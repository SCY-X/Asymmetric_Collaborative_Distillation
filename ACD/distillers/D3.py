import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller

def d3_loss(student_features, teacher_features, topk, alpha, beta, gamma): 

    batch_size = student_features.shape[0]

    # Normalize student and teacher features
    student_features = F.normalize(student_features, p=2, dim=1)
    teacher_features = F.normalize(teacher_features, p=2, dim=1)

    # Compute similarity matrices
    teacher_similarity = teacher_features.double().mm(teacher_features.double().t())
    cross_similarity = student_features.double().mm(teacher_features.double().t())

    # Get sorted teacher similarity and corresponding cross similarity
    teacher_topk_values, sorted_indices = torch.sort(teacher_similarity, dim=1, descending=True)
    student_topk_values = torch.gather(cross_similarity, 1, sorted_indices)

    # Feature distillation loss (alpha term)
    fd_loss = alpha * torch.norm(teacher_topk_values[:, 0] - student_topk_values[:, 0], p=2) / batch_size

   

    # Extract top-k similarities (excluding the highest)
    student_distances = student_topk_values[:, 1:topk]
    teacher_distances = teacher_topk_values[:, 1:topk]
    
   
    # Compute pairwise difference matrices
    student_diff_matrix = student_distances.unsqueeze(1) - student_distances.unsqueeze(2)
    teacher_diff_matrix = teacher_distances.unsqueeze(1) - teacher_distances.unsqueeze(2)

    # Flatten the difference matrices
    student_diff_flat = student_diff_matrix.view(batch_size, -1)
    teacher_diff_flat = teacher_diff_matrix.view(batch_size, -1)
    
    # Avoid division by zero
    teacher_diff_flat[teacher_diff_flat == 0] = 1

    # Compute hard and simple weights
    hard_weights = (student_diff_flat / teacher_diff_flat).detach()
    simple_weights = (student_diff_flat / teacher_diff_flat).detach()

    hard_weights[hard_weights >= 0] = 0
    hard_weights[hard_weights < 0] = 1

    simple_weights[simple_weights <= 0] = 0
    simple_weights[simple_weights > 0] = 1

    # Avoid division by zero in student differences
    student_diff_flat[student_diff_flat == 0] = 1

    # Compute weighted result matrices
    hard_loss_matrix = hard_weights * ((student_diff_flat - teacher_diff_flat) / (0.1 + teacher_diff_flat.abs()))
    simple_loss_matrix = simple_weights * ((student_diff_flat - teacher_diff_flat) / (0.1 + teacher_diff_flat.abs()))

    # Relation KD loss (beta and gamma terms)
    hard_rd_loss = beta * torch.mean(torch.norm(hard_loss_matrix, p=2, dim=1)) / (topk-1)

    simple_rd_loss = gamma * torch.mean(torch.norm(simple_loss_matrix, p=2, dim=1)) / (topk-1)

    return fd_loss +  hard_rd_loss + simple_rd_loss

   
def vanillakd_loss(logits_student_in, logits_teacher_in, temperature):
    log_pred_student = F.log_softmax(logits_student_in / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher_in / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd



class D3(Distiller):
    """D3still: Decoupled Differential Distillation for Asymmetric Image Retrieval. CVPR2024"""

    def __init__(self, student, teacher, cfg):
        super(D3, self).__init__(student, teacher, cfg)

        self.topk = cfg.D3.TOPK
        self.alpha = cfg.D3.ALPHA
        self.beta = cfg.D3.BETA
        self.gamma = cfg.D3.GAMMA

        self.kd_loss_weight = cfg.D3.KD_WEIGHT

        self.acld_alpha = cfg.DISTILLER.TEACHER_ALPHA
        self.acld_beta = cfg.DISTILLER.TEACHER_BETA
        self.acld_t = cfg.DISTILLER.TEACHER_TEMPERATURE

    
    def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):

        logits_teacher, feature_teacher = self.teacher(image)

        ce_loss = self.ce_loss_weight * self.ce_loss(logits_teacher, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_teacher["pooled_feat"], target) 

        
        kd_logits_student, kd_feature_student = self.student(kd_student_image)
        kd_logits_teacher, kd_feature_teacher = self.teacher(kd_teacher_image)


    
        kd_feature_loss = self.kd_loss_weight * d3_loss(kd_feature_student["retrieval_feat"], kd_feature_teacher["retrieval_feat"],
                                                self.topk, self.alpha, self.beta, self.gamma)
 
        kd_logit_st = self.teacher.fc.classifier((kd_feature_teacher["retrieval_feat"] + kd_feature_student["retrieval_feat"]) / 2.0)    
        
        kd_logit_loss = self.acld_alpha * vanillakd_loss(kd_logits_student, kd_logits_teacher, self.acld_t) + self.acld_beta * vanillakd_loss(kd_logits_student, kd_logit_st, self.acld_t / 2) 
       
        
        kd_loss = kd_feature_loss + kd_logit_loss

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_teacher, losses_dict


