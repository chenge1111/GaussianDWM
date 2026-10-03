from __future__ import annotations
import torch
from torch import nn
from torchvision import models


class ResnetEncoder(nn.Module):
    """Monodepth/PackNet-compatible ResNet feature interface and state key layout."""
    def __init__(self, num_layers: int, pretrained: bool, num_input_images: int = 1):
        super().__init__()
        if num_layers not in {18,34} or num_input_images != 1:
            raise ValueError("DrivingForward adapter supports ResNet18/34 with one RGB input")
        ctor = models.resnet18 if num_layers == 18 else models.resnet34
        weights = (models.ResNet18_Weights.DEFAULT if num_layers == 18 else
                   models.ResNet34_Weights.DEFAULT) if pretrained else None
        self.encoder = ctor(weights=weights)
        self.num_ch_enc = [64,64,128,256,512]

    def forward(self, image):
        x = (image-0.45)/0.225
        x = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        outputs = [x]
        x = self.encoder.layer1(self.encoder.maxpool(x)); outputs.append(x)
        x = self.encoder.layer2(x); outputs.append(x)
        x = self.encoder.layer3(x); outputs.append(x)
        x = self.encoder.layer4(x); outputs.append(x)
        return outputs


def axis_angle_to_matrix(angle: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(angle,dim=-1,keepdim=True)
    x,y,z = angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(*angle.shape[:-1],3,3)
    a = torch.sinc(theta/torch.pi)[...,None]
    b = (0.5*torch.sinc(theta/(2*torch.pi))**2)[...,None]
    identity = torch.eye(3,device=angle.device,dtype=angle.dtype)
    return identity + a*skew + b*(skew@skew)
