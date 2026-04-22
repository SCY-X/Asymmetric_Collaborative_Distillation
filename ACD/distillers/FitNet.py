import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


def vanillakd_loss(logits_student_in, logits_teacher_in, temperature):
    log_pred_student = F.log_softmax(logits_student_in / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher_in / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

class FitNet(Distiller):
    """FitNets: Hints for Thin Deep Nets"""

    def __init__(self, student, teacher, cfg):
        super(FitNet, self).__init__(student, teacher, cfg)
      
        self.kd_loss = nn.MSELoss()
        self.kd_loss_weight = cfg.FITNET.KD_WEIGHT

        self.acld_alpha = cfg.DISTILLER.TEACHER_ALPHA
        self.acld_beta = cfg.DISTILLER.TEACHER_BETA
        self.acld_t = cfg.DISTILLER.TEACHER_TEMPERATURE
         

    def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):

        logits_teacher, feature_teacher = self.teacher(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_teacher, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_teacher["pooled_feat"], target)

        kd_logits_student, kd_feature_student = self.student(kd_student_image)
        kd_logits_teacher, kd_feature_teacher = self.teacher(kd_teacher_image)

        kd_feature_loss = self.kd_loss_weight * self.kd_loss(kd_feature_student["retrieval_feat"], kd_feature_teacher["retrieval_feat"])

        kd_logit_st = self.teacher.fc.classifier((kd_feature_teacher["retrieval_feat"] + kd_feature_student["retrieval_feat"]) / 2.0)   
        kd_logit_loss = self.acld_alpha * vanillakd_loss(kd_logits_student, kd_logits_teacher, self.acld_t) + self.acld_beta * vanillakd_loss(kd_logits_student, kd_logit_st, self.acld_t / 2) 
       

        kd_loss = kd_feature_loss + kd_logit_loss

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        } 
        return logits_teacher, losses_dict