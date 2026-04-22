import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class DownSampling_Pooling(nn.Module):
    def __init__(self, size):
        super(DownSampling_Pooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(size)
      
    def forward(self, x):
        x = self.pooling(x)
        x_flatten = torch.flatten(x, start_dim=2, end_dim=3)

        x_flatten = x_flatten.permute((0, 2, 1))
        m, b, c = x_flatten.shape[0], x_flatten.shape[1], x_flatten.shape[2]

        x_flatten = torch.reshape(x_flatten, (m * b, c))
        return x_flatten


def vanillakd_loss(logits_student_in, logits_teacher_in, temperature):
    log_pred_student = F.log_softmax(logits_student_in / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher_in / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class UGD(Distiller):
    """Unambiguous granularity distillation for asymmetric image retrieval. Neural Networks 2025"""

    def __init__(self, student, teacher, cfg):
        super(UGD, self).__init__(student, teacher, cfg)
      
        
        self.distillation_layer = cfg.UGD.DISTILLATION_LAYER
        self.alpha = cfg.UGD.ALPHA 
        self.beta = cfg.UGD.BETA
        self.kd_loss_weight = cfg.UGD.KD_WEIGHT

        pooling_size = cfg.INPUT.STUDENT_SIZE_TRAIN[0] // 16
      
        self.downsampling_pooling = DownSampling_Pooling((pooling_size, pooling_size))
        self.downsampling_pooling_half = DownSampling_Pooling((pooling_size // 2, pooling_size // 2))
        self.num_bottleneck = 512

        model_channel_num = {"ResNet18": [64, 128, 256, 512],
                            "MobileNetV3_Small": [16, 24, 48, 576],
                            "ResNet101": [256, 512, 1024, 2048]}
        

        self.acld_alpha = cfg.DISTILLER.TEACHER_ALPHA
        self.acld_beta = cfg.DISTILLER.TEACHER_BETA
        self.acld_t = cfg.DISTILLER.TEACHER_TEMPERATURE

        
        channel_num = model_channel_num[cfg.DISTILLER.STUDENT_NAME][self.distillation_layer - 1]

        self.projector = nn.Sequential(*[nn.BatchNorm1d(channel_num), nn.Linear(channel_num, self.num_bottleneck, bias=False), nn.BatchNorm1d(self.num_bottleneck)])
        self.projector.apply(weights_init_kaiming)

        self.projector_half = nn.Sequential(*[nn.BatchNorm1d(channel_num), nn.Linear(channel_num, self.num_bottleneck, bias=False), nn.BatchNorm1d(self.num_bottleneck)])
        self.projector_half.apply(weights_init_kaiming)
    
    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.projector.named_parameters()) + list(self.projector_half.named_parameters())

    def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):

        logits_teacher, feature_teacher = self.teacher(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_teacher, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_teacher["pooled_feat"], target)


        kd_logits_student, kd_feature_student = self.student(kd_student_image)
        kd_logits_teacher, kd_feature_teacher = self.teacher(kd_teacher_image)

        kd_feature_loss = self.kd_loss_weight * self.ugd_loss(kd_feature_teacher, kd_feature_student, kd_target)

        kd_logit_st = self.teacher.fc.classifier((kd_feature_teacher["retrieval_feat"] + kd_feature_student["retrieval_feat"]) / 2.0)   
        kd_logit_loss = self.acld_alpha * vanillakd_loss(kd_logits_student, kd_logits_teacher, self.acld_t) + self.acld_beta * vanillakd_loss(kd_logits_student, kd_logit_st, self.acld_t / 2) 
       
        kd_loss = kd_feature_loss + kd_logit_loss

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_teacher, losses_dict


    def ugd_loss(self, kd_feature_teacher, kd_feature_student, target):

        t_local = self.downsampling_pooling(kd_feature_teacher["feats"][-1])
        t_local_score, t_local_feat = self.teacher.fc(t_local)


        t_local_half = self.downsampling_pooling_half(kd_feature_teacher["feats"][-1])
        t_local_half_score, t_local_half_feat = self.teacher.fc(t_local_half)


        s_local = self.downsampling_pooling(kd_feature_student["feats"][self.distillation_layer - 1])
        s_local_feat = self.projector(s_local)

        
        s_local_half =  self.downsampling_pooling_half(kd_feature_student["feats"][self.distillation_layer - 1])
        s_local_half_feat = self.projector_half(s_local_half)
        
        
        s_feat = F.normalize(kd_feature_student["retrieval_feat"], p=2, dim=1)
        t_feat = F.normalize(kd_feature_teacher["retrieval_feat"], p=2, dim=1)
        
        f_loss = torch.norm(t_feat - s_feat, p=2, dim=1).mean()

        t_local_feat =  F.normalize(t_local_feat, p=2, dim=1)
        t_local_half_feat = F.normalize(t_local_half_feat, p=2, dim=1)
    
        s_local_feat =  F.normalize(s_local_feat, p=2, dim=1)
        s_local_half_feat = F.normalize(s_local_half_feat, p=2, dim=1)

       
        local_target = target.repeat_interleave(int(s_local_feat.size(0) / len(target)))
        local_half_target = target.repeat_interleave(int(s_local_half_feat.size(0) / len(target)))

        
        t_local_predict = torch.argmax(t_local_score, dim=1)
        t_local_half_predict = torch.argmax(t_local_half_score, dim=1)
        

        local_true_mask = t_local_predict  ==  local_target
    
        local_true_half_mask = t_local_half_predict ==  local_half_target
    
        if local_true_mask.any():
            local_true_loss = torch.norm(t_local_feat - s_local_feat, p=2, dim=1)[local_true_mask].mean()
          
        else:
            local_true_loss = torch.Tensor([0.0]).cuda()

        if local_true_half_mask.any():
            local_true_half_loss = torch.norm(t_local_half_feat - s_local_half_feat, p=2, dim=1)[local_true_half_mask].mean()
          
        else:
            local_true_half_loss = torch.Tensor([0.0]).cuda()
 
        loss = self.alpha * f_loss + self.beta * (local_true_loss + local_true_half_loss)

        return loss