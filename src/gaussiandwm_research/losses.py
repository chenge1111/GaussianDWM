from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def masked_l1(pred,gt,mask):
    valid = mask.expand_as(gt) & torch.isfinite(gt)
    return (pred-gt.nan_to_num()).abs()[valid].mean() if valid.any() else pred.sum()*0


class PerceptualLoss(nn.Module):
    """Frozen ImageNet VGG16 feature loss; gradients to predicted RGB are retained."""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16,VGG16_Weights
        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23].eval().requires_grad_(False)
        self.register_buffer("mean",torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer("std",torch.tensor([.229,.224,.225])[None,:,None,None])

    def train(self,mode=True):
        super().train(mode)
        self.features.eval()
        return self

    def forward(self,pred,target):
        pred = pred.flatten(0,1).float()
        target = target.flatten(0,1).float()
        pred = F.interpolate(pred,(224,224),mode="bilinear",align_corners=False)
        target = F.interpolate(target,(224,224),mode="bilinear",align_corners=False)
        return F.l1_loss(self.features((pred-self.mean)/self.std),
                         self.features((target-self.mean)/self.std))
