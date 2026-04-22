import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


def csd_loss(student_features, teacher_features, top_k, temp_student, temp_teacher):
    # Normalize features with L2 normalization
    student_features = F.normalize(student_features, p=2, dim=1)
    teacher_features = F.normalize(teacher_features, p=2, dim=1)

    # Compute similarity matrix for teacher features
    teacher_similarity = teacher_features.double().mm(teacher_features.double().t())
    # Compute cross-similarity matrix between student and teacher features
    cross_similarity = student_features.double().mm(teacher_features.double().t())

    # Get the top-k indices of teacher similarity for retrieval
    teacher_topk_values, teacher_topk_indices = torch.sort(teacher_similarity, dim=1, descending=True)
    topk_gallery_indices = teacher_topk_indices[:, :top_k]

    # Gather top-k similarity scores for student predictions
    student_topk_scores = torch.gather(cross_similarity, 1, topk_gallery_indices)

    # Extract top-k similarity scores from teacher similarity matrix
    teacher_topk_scores = teacher_topk_values[:, :top_k]

    # Compute probability distributions for student and teacher
    student_probs = F.log_softmax(student_topk_scores / temp_student, dim=1)
    teacher_probs = F.softmax(teacher_topk_scores / temp_teacher, dim=1)

    # Compute KL divergence loss
    loss = F.kl_div(student_probs, teacher_probs.detach(), reduction='sum') / teacher_features.shape[0]

    return loss

def vanillakd_loss(logits_student_in, logits_teacher_in, temperature):
    log_pred_student = F.log_softmax(logits_student_in / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher_in / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

class CSD(Distiller):
    """Contextual Similarity Distillation for Asymmetric Image Restrieval. CVPR2023"""

    """
    The original paper adopts an offline approach to extracting gallery features from the entire training set. 
    However, through practical experiments, it was found that directly extracting gallery features for each batch achieves better performance. 
    Therefore, in this project, I chose to use the latter approach. It is important to note that this modification is based on 
    personal experimentation and is not directly related to the original paper. The effectiveness of this approach may vary 
    depending on the specific dataset or experimental settings.
    """

    def __init__(self, student, teacher, cfg):
        super(CSD, self).__init__(student, teacher, cfg)

        self.k = cfg.CSD.TOPK
        self.tq = cfg.CSD.TEMPERATURE_QUERY
        self.tg = cfg.CSD.TEMPERATURE_GALLERY
        self.kd_loss_weight = cfg.CSD.KD_WEIGHT

        self.acld_alpha = cfg.DISTILLER.TEACHER_ALPHA
        self.acld_beta = cfg.DISTILLER.TEACHER_BETA
        self.acld_t = cfg.DISTILLER.TEACHER_TEMPERATURE
      

    def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):

        logits_teacher, feature_teacher = self.teacher(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_teacher, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_teacher["pooled_feat"], target)

        kd_logits_student, kd_feature_student = self.student(kd_student_image)
        kd_logits_teacher, kd_feature_teacher = self.teacher(kd_teacher_image)

        kd_feature_loss = self.kd_loss_weight * csd_loss(kd_feature_student["retrieval_feat"],
                                                 kd_feature_teacher["retrieval_feat"], 
                                                 self.k,
                                                 self.tq,
                                                 self.tg)
        
        kd_logit_st = self.teacher.fc.classifier((kd_feature_teacher["retrieval_feat"] + kd_feature_student["retrieval_feat"]) / 2.0)   
        kd_logit_loss = self.acld_alpha * vanillakd_loss(kd_logits_student, kd_logits_teacher, self.acld_t) + self.acld_beta * vanillakd_loss(kd_logits_student, kd_logit_st, self.acld_t / 2) 
       

        kd_loss = kd_feature_loss + kd_logit_loss

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_teacher, losses_dict
