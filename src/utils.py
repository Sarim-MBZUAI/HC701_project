from torchmetrics.classification.accuracy import Accuracy
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, CyclicLR
from torchmtlr import mtlr_neg_log_likelihood, mtlr_survival, mtlr_risk
import numpy as np
from scipy.spatial import cKDTree
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
import torch
from torch import nn
from torchmtlr import MTLR

def make_optimizer(opt_cls, model, **kwargs):
    """Creates a PyTorch optimizer for MTLR training."""
    params_dict = dict(model.named_parameters())
    weights = [v for k, v in params_dict.items() if "mtlr" not in k and "bias" not in k]
    biases = [v for k, v in params_dict.items() if "bias" in k]
    mtlr_weights = [v for k, v in params_dict.items() if "mtlr_weight" in k]
    # Don't use weight decay on the biases and MTLR parameters, which have
    # their own separate L2 regularization
    optimizer = opt_cls([
        {"params": weights},
        {"params": biases, "weight_decay": 0.},
        {"params": mtlr_weights, "weight_decay": 0.},], 
        **kwargs)
    return optimizer

def dice(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)

    return score.mean()

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss
    
def hausdorff_distance(image0, image1):

    """Code copied from 
    https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/set_metrics.py#L7-L54
    for compatibility reason with python 3.6
    """
    a_points = np.transpose(np.nonzero(image0.cpu()))
    b_points = np.transpose(np.nonzero(image1.cpu()))

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
               max(cKDTree(b_points).query(a_points, k=1)[0]))

def training_one_step(model, criterion, C1,loss_gamma,  batch):
    (sample, clin_var), y, labels = batch

    pred_mask, risk_out = model((sample['input'], clin_var))

    y, target_mask, risk_out = y.to("cuda"), sample['target_mask'].to("cuda"), risk_out.to("cuda")
    
    loss_mtlr = mtlr_neg_log_likelihood(risk_out, y.float(), model, C1, average=True)
    true_time   = torch.cat([labels["time"]]).cpu()
    true_event  = torch.cat([labels["event"]]).cpu()
    risk_out = risk_out.cpu()
    pred_risk = mtlr_risk(risk_out).detach().numpy()

    try:
        ci_event  = concordance_index(true_time, -pred_risk, event_observed=true_event)
    except:
        ci_event = 0

    loss_mask = criterion(pred_mask, target_mask)
    mean_dice = dice(pred_mask, sample['target_mask'].to("cuda"))

    loss = (1-loss_gamma)*loss_mtlr + loss_gamma*loss_mask

        

    
    return loss, loss_mtlr, loss_mask, mean_dice, ci_event
    
def validation_one_step(model, criterion, C1,loss_gamma,  batch):
    model.eval()
    with torch.no_grad():
        loss, loss_mtlr, loss_mask, mean_dice, ci_event = training_one_step(model, criterion, C1,loss_gamma,  batch)

    return loss, loss_mtlr, loss_mask, mean_dice, ci_event