import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


def normalize_within_layers(all_batch_scores, block_names):
    block_layer_map = {name: name.split('.')[0] for name in block_names}
    layer_groups = {}
    for i, name in enumerate(block_names):
        layer = block_layer_map[name]
        layer_groups.setdefault(layer, []).append(i)

   
    norm_scores = torch.zeros_like(all_batch_scores)
    for layer, indices in layer_groups.items():
        scores = all_batch_scores[indices]
        normed = torch.softmax(scores, dim=0) * len(scores)
        norm_scores[indices] = normed

    return norm_scores


# #10-22
def analyze_layer_importance(model, dataloader, device, cfg):

    ratio = cfg.DISTILLER.TEACHER_RATIO
    cross_resolution = cfg.DISTILLER.CROSS_RESOLUTION
   

    # if cfg.DATASETS.NAMES == "CUB200" and  cfg.INPUT.TEACHER_SIZE_TRAIN == [512, 512]: 
    #     target_prefixes = ['layer3', 'layer4']
    # else:
    target_prefixes = ['layer1', 'layer2', 'layer3', 'layer4']

    model.eval()

    with torch.no_grad():

        all_block_scores_tt = []
        all_block_scores_ts = []

        block_names = [] 
        for module_name, module in model.teacher.named_modules():
            if any(module_name.startswith(prefix) for prefix in target_prefixes):
                if hasattr(module, 'get_block_feature'):   
                    block_names.append(module_name)
        
        
        for stu_img, tea_img, target in dataloader:
           
            stu_img = stu_img.to(device)
            tea_img = tea_img.to(device)
            target = target.to(device)

            _, feat_tea = model.teacher(tea_img)

            block_features_tt = []
            for module_name, module in model.teacher.named_modules():
                if any(module_name.startswith(prefix) for prefix in target_prefixes):
                    if hasattr(module, 'get_block_feature'):
                        block_feat = F.normalize(module.get_block_feature(), p=2, dim=1)
                        assert block_feat.dim() == 2, f"{module_name} output must be 2D, got {block_feat.shape}"
                        block_features_tt.append(block_feat)

         
            logit_tea_stu, feat_tea_stu = model.teacher(stu_img)

            # 计算预测是否正确，得到掩码
            pred_correct_mask = (logit_tea_stu.argmax(dim=1) == target)  # [B], bool

            block_features_ts = []
            for module_name, module in model.teacher.named_modules():
                if any(module_name.startswith(prefix) for prefix in target_prefixes):
                    if hasattr(module, 'get_block_feature'):
                        block_feat = F.normalize(module.get_block_feature()[pred_correct_mask], p=2, dim=1)
                        assert block_feat.dim() == 2, f"{module_name} output must be 2D, got {block_feat.shape}"
                        block_features_ts.append(block_feat)
           

            _, feat_stu = model.student(stu_img)

            norm_tea_feat = F.normalize(feat_tea["retrieval_feat"], p=2, dim=1)
            norm_stu_feat = F.normalize(feat_stu["retrieval_feat"][pred_correct_mask], p=2, dim=1)

            sim_tt = torch.mm(norm_tea_feat, norm_tea_feat.T)
            sim_ss = torch.mm(norm_stu_feat, norm_stu_feat.T)

          
            blockwise_tt_diff, blockwise_ts_diff = [], []
        
            for layer_idx, (block_feat_tt, block_feat_ts) in enumerate(zip(block_features_tt, block_features_ts)):
                
                sim_block_tt = torch.mm(block_feat_tt, block_feat_tt.T)
                sim_block_ts = torch.mm(block_feat_ts, block_feat_ts.T)
             

                diff_tt = torch.norm(sim_tt - sim_block_tt, p=2, dim=1).mean()
                diff_ts = torch.norm(sim_ss - sim_block_ts, p=2, dim=1).mean()
        
                blockwise_tt_diff.append(diff_tt)
                blockwise_ts_diff.append(diff_ts)
            
        
            blockwise_tt_diff = torch.tensor(blockwise_tt_diff).to(device)
            blockwise_ts_diff = torch.tensor(blockwise_ts_diff).to(device)

         
            tt_diff_tensor = blockwise_tt_diff[1:] - blockwise_tt_diff[:-1]
            ts_diff_tensor = blockwise_ts_diff[1:] - blockwise_ts_diff[:-1]

            all_block_scores_tt.append(-tt_diff_tensor)
            all_block_scores_ts.append(-ts_diff_tensor)
            

    
        avg_scores_tt = torch.stack(all_block_scores_tt, dim=0).mean(dim=0) 
        avg_scores_ts = torch.stack(all_block_scores_ts, dim=0).mean(dim=0)
       

        norm_scores_tt = normalize_within_layers(avg_scores_tt, block_names[1:])
        norm_scores_ts = normalize_within_layers(avg_scores_ts, block_names[1:])

        num_keep = int(len(avg_scores_tt) * ratio)
     
        unimport_index_tt = torch.argsort(norm_scores_tt)[:num_keep]
        unimport_index_ts = torch.argsort(norm_scores_ts)[:num_keep]


        set_diff_tt = set(unimport_index_tt.tolist())
        set_diff_ts = set(unimport_index_ts.tolist())

        # 根据 cross_resolution 决定活跃集合
        active_sets = [set_diff_tt]
        if cross_resolution:
            active_sets.append(set_diff_ts)

        # 计算交集，如果没有集合则返回空集合
        intersection = set.intersection(*active_sets) if active_sets else set()

        # 非交集部分
        non_intersection = set_diff_tt - intersection
    

        inter_unimportant_block_names = [block_names[1:][i] for i in intersection]
        noninter_unimportant_block_names = [block_names[1:][i] for i in non_intersection]

        
        # === 返回交集 block 名 + 非交集 block 名 ===
        return inter_unimportant_block_names,  noninter_unimportant_block_names


def swin_transformer_normalize_within_layers(all_batch_scores, block_names):
  
    block_layer_map = {name: name.split('.')[2] for name in block_names}
   
    layer_groups = {}
    for i, name in enumerate(block_names):
        layer = block_layer_map[name]
        layer_groups.setdefault(layer, []).append(i)

    norm_scores = torch.zeros_like(all_batch_scores)
    for layer, indices in layer_groups.items():
        scores = all_batch_scores[indices]
        normed = torch.softmax(scores, dim=0) * len(scores)
        norm_scores[indices] = normed
        
    return norm_scores



def swin_transformer_analyze_layer_importance(model, dataloader, device, cfg):

    ratio = cfg.DISTILLER.TEACHER_RATIO
    cross_resolution = cfg.DISTILLER.CROSS_RESOLUTION
   

    target_prefixes = ['model.layers.0', 'model.layers.1', 'model.layers.2', 'model.layers.3']

    model.eval()

    with torch.no_grad():

        all_block_scores_tt = []

        block_names = [] 
        for module_name, module in model.teacher.named_modules():
        
            if any(module_name.startswith(prefix) for prefix in target_prefixes):
                if hasattr(module, 'get_block_feature'):   
                    block_names.append(module_name)

        for stu_img, tea_img, target in dataloader:
           
            stu_img = stu_img.to(device)
            tea_img = tea_img.to(device)
            target = target.to(device)

            _, feat_tea = model.teacher(tea_img)

            block_features_tt = []
            for module_name, module in model.teacher.named_modules():
                if any(module_name.startswith(prefix) for prefix in target_prefixes):
                    if hasattr(module, 'get_block_feature'):
                        block_feat = F.normalize(module.get_block_feature(), p=2, dim=1)
                        assert block_feat.dim() == 2, f"{module_name} output must be 2D, got {block_feat.shape}"
                        block_features_tt.append(block_feat)

         
            norm_tea_feat = F.normalize(feat_tea["retrieval_feat"], p=2, dim=1)

            sim_tt = torch.mm(norm_tea_feat, norm_tea_feat.T) 
         
            blockwise_tt_diff = []
        
            for layer_idx, block_feat_tt in enumerate(block_features_tt):
                
                sim_block_tt = torch.mm(block_feat_tt, block_feat_tt.T)
            
                diff_tt = torch.norm(sim_tt - sim_block_tt, p=2, dim=1).mean()
              
                blockwise_tt_diff.append(diff_tt)
            
        
            blockwise_tt_diff = torch.tensor(blockwise_tt_diff).to(device)
         
            tt_diff_tensor = blockwise_tt_diff[1:] - blockwise_tt_diff[:-1]

            all_block_scores_tt.append(-tt_diff_tensor)

            
    
        avg_scores_tt = torch.stack(all_block_scores_tt, dim=0).mean(dim=0) 
     
        norm_scores_tt = swin_transformer_normalize_within_layers(avg_scores_tt, block_names[1:])
    

        num_keep = int(len(avg_scores_tt) * ratio)
     
        unimport_index_tt = torch.argsort(norm_scores_tt)[:num_keep]

        set_diff_tt = set(unimport_index_tt.tolist())

        # 根据 cross_resolution 决定活跃集合
        active_sets = [set_diff_tt]

        if cross_resolution:
            active_sets.append([])

        # 计算交集，如果没有集合则返回空集合
        intersection = set.intersection(*active_sets) if active_sets else set()

        # 非交集部分
        non_intersection = set_diff_tt - intersection
    
        inter_unimportant_block_names = [block_names[1:][i] for i in intersection]
        noninter_unimportant_block_names = [block_names[1:][i] for i in non_intersection]

        # === 返回交集 block 名 + 非交集 block 名 ===
        return inter_unimportant_block_names,  noninter_unimportant_block_names