import torch
import logging


def make_optimizer(cfg, distiller, inter_unimportant_block_names, noninter_unimportant_block_names):
    logger = logging.getLogger("Asymmetric_Image_Retrieval.train")
    params = []
    
    parameter_source = distiller.module if torch.cuda.device_count() > 1 else distiller 

    for key, value in parameter_source.get_learnable_parameters():

        if not value.requires_grad:
            continue
        
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if cfg.DISTILLER.TEACHER_NAME == "Swin_Transformer_V2_Small":
            layer_prefix = ".".join(key.split(".")[:5])
        else:
            layer_prefix = ".".join(key.split(".")[:2])
 
        if layer_prefix in inter_unimportant_block_names:
            lr = cfg.SOLVER.BASE_LR

        elif layer_prefix in noninter_unimportant_block_names:
            #weight_decay = cfg.SOLVER.UNIMPORTANT_LR_TIMES * cfg.SOLVER.WEIGHT_DECAY
            lr = cfg.SOLVER.UNIMPORTANT_LR_TIMES * cfg.SOLVER.BASE_LR
       
        else:
            #weight_decay = cfg.SOLVER.UNIMPORTANT_LR_TIMES**2  * cfg.SOLVER.WEIGHT_DECAY
            lr = cfg.SOLVER.UNIMPORTANT_LR_TIMES**2 * cfg.SOLVER.BASE_LR
       

        if "bias" in key:
            lr = lr * cfg.SOLVER.BIAS_LR_FACTOR
            # weight_decay = weight_decay * cfg.SOLVER.WEIGHT_DECAY_BIAS

        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key:
                lr = lr * cfg.SOLVER.FC_LR_TIMES
                # weight_decay = weight_decay * cfg.SOLVER.FC_LR_TIMES
                logger.info('Using {} times learning rate for fc'.format(cfg.SOLVER.FC_LR_TIMES))

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

   
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    return optimizer



