import torch 
import torch.nn as nn
import torch.nn.functional as F

def js_loss_compute(pred, soft_targets, reduce=True):
    
    pred_softmax = F.softmax(pred, dim=1)
    targets_softmax = F.softmax(soft_targets, dim=1)
    mean = (pred_softmax + targets_softmax) / 2
    kl_1 = F.kl_div(F.log_softmax(pred, dim=1), mean, reduce=False)
    kl_2 = F.kl_div(F.log_softmax(soft_targets, dim=1), mean, reduce=False)
    js = (kl_1 + kl_2) / 2 
    
    if reduce:
        return torch.mean(torch.sum(js, dim=1))
    else:
        return torch.sum(js, 1)