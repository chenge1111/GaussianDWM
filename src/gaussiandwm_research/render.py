from __future__ import annotations
import torch
from torch import nn
from .scene import GaussianScene


class GaussianRenderer(nn.Module):
    """gsplat alpha compositing for RGB, expected metric depth and CLIP features.

    No detach/no_grad: downstream losses flow to geometry, opacity, color and
    language. Camera calibration is fixed. Feature channels are rendered in
    chunks by gsplat, with autograd enabled.
    """
    def __init__(self, near_plane: float = 0.1, far_plane: float = 100):
        super().__init__()
        self.near_plane,self.far_plane = near_plane,far_plane

    def forward(self, scene: GaussianScene, intrinsics: torch.Tensor,
                camera_to_ego: torch.Tensor, height: int, width: int,
                render_language: bool = False, seconds: float = 0) -> dict[str,torch.Tensor]:
        from gsplat import rasterization
        scene = scene.at_time(seconds)
        rgbs,depths,alphas,semantic = [],[],[],[]
        # Per-scene calls support gsplat 1.5.x without newer batched APIs.
        for bi in range(scene.xyz.shape[0]):
            valid = scene.valid[bi]
            args = dict(means=scene.xyz[bi,valid].float(),quats=scene.rotations[bi,valid].float(),
                        scales=scene.log_scales[bi,valid].float().exp(),
                        opacities=scene.opacity[bi,valid,0].float(),
                        viewmats=torch.linalg.inv(camera_to_ego[bi].float()),Ks=intrinsics[bi].float(),
                        width=width,height=height,near_plane=self.near_plane,far_plane=self.far_plane,
                        packed=True,channel_chunk=32)
            result,alpha,_ = rasterization(colors=scene.colors[bi,valid].float(),
                                          render_mode="RGB+ED",**args)
            rgbs.append(result[...,:3].permute(0,3,1,2))
            depths.append(result[...,3:4].permute(0,3,1,2))
            alphas.append(alpha.permute(0,3,1,2))
            if render_language:
                features,_,_ = rasterization(colors=scene.language[bi,valid].float(),
                                             render_mode="RGB",**args)
                semantic.append((features/alpha.clamp_min(1e-6)).permute(0,3,1,2))
        out = {"rgb":torch.stack(rgbs),"depth":torch.stack(depths),"alpha":torch.stack(alphas)}
        if semantic:
            out["language"] = torch.stack(semantic)
        return out
