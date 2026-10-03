from __future__ import annotations

from dataclasses import replace
import torch
import torch.nn.functional as F
from .scene import GaussianScene


def project(xyz: torch.Tensor, camera_to_ego: torch.Tensor,
            intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ego_to_cam = torch.linalg.inv(camera_to_ego.float())
    pts = torch.einsum("bvij,bnj->bvni", ego_to_cam[..., :3, :3], xyz.float())
    pts = pts + ego_to_cam[..., :3, 3].unsqueeze(-2)
    pixels = torch.einsum("bvij,bvnj->bvni", intrinsics.float(), pts)
    uv = pixels[..., :2] / pixels[..., 2:].clamp_min(1e-5)
    return uv, pts[..., 2]


@torch.no_grad()
def estimate_occlusion(scene: GaussianScene, intrinsics: torch.Tensor,
                       camera_to_ego: torch.Tensor, height: int, width: int,
                       grid_stride: int = 4, depth_margin: float = 0.5,
                       threshold: float = 0.5, chunk_size: int = 8192) -> torch.Tensor:
    """Nine footprint probes behind a foreground z-buffer estimate coverage.

    A Gaussian is occluded when > threshold of its footprint is behind another
    Gaussian in every camera that observes it. Opacity is NOT an occlusion label.
    This is a geometric proxy, not ground-truth occlusion. Off-screen points are
    not marked occluded. Hard visibility/instance decisions are inference metadata.
    """
    uv, depths = project(scene.xyz, camera_to_ego, intrinsics)
    gh, gw = (height + grid_stride-1)//grid_stride, (width + grid_stride-1)//grid_stride
    b, v, n = depths.shape
    visible = (depths > 0.1) & (uv[..., 0] >= 0) & (uv[..., 0] < width) & (
        uv[..., 1] >= 0) & (uv[..., 1] < height) & scene.valid[:, None]
    radius = (scene.log_scales.exp().amax(-1)[:, None] * intrinsics[..., 0, 0, None]
              / depths.clamp_min(0.1)).clamp(grid_stride, 32)
    probes = torch.tensor([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],
                           [1,-1],[1,0],[1,1]], device=uv.device)
    ratios = []
    for bi in range(b):
        view_ratios = []
        for vi in range(v):
            zbuffer = torch.full((gh*gw,), float("inf"), device=uv.device)
            # Scatter footprint depth as well as centers; no NxN overlap matrix.
            for start in range(0, n, chunk_size):
                sl = slice(start, start+chunk_size)
                xy = uv[bi,vi,sl,None,:] + radius[bi,vi,sl,None,None]*probes
                xy = (xy/grid_stride).long()
                keep = visible[bi,vi,sl,None] & (xy[...,0]>=0) & (xy[...,0]<gw) & (
                    xy[...,1]>=0) & (xy[...,1]<gh) & (scene.opacity[bi,sl,0,None]>0.05)
                idx = xy[...,1]*gw+xy[...,0]
                zs = depths[bi,vi,sl,None].expand_as(idx)
                zbuffer.scatter_reduce_(0, idx[keep], zs[keep], reduce="amin", include_self=True)
            xy = uv[bi,vi,:,None,:] + radius[bi,vi,:,None,None]*probes
            xy = (xy/grid_stride).long()
            valid_probe = (xy[...,0]>=0)&(xy[...,0]<gw)&(xy[...,1]>=0)&(xy[...,1]<gh)
            idx = (xy[...,1].clamp(0,gh-1)*gw + xy[...,0].clamp(0,gw-1))
            behind = (depths[bi,vi,:,None] > zbuffer[idx]+depth_margin) & valid_probe
            ratio = behind.float().sum(-1)/valid_probe.sum(-1).clamp_min(1)
            view_ratios.append(ratio)
        ratios.append(torch.stack(view_ratios))
    coverage = torch.stack(ratios)
    return ((coverage > threshold) | ~visible).all(1) & visible.any(1) & scene.valid


def enhance_semantics(scene: GaussianScene, occluded: torch.Tensor,
                      neighbors: int = 8, strength: float = 1.0,
                      chunk_size: int = 256) -> GaussianScene:
    """Fill only known-instance occluded rows from visible same-instance neighbors.

    Never infer object identity from opacity or mix all unknown/background points.
    Feature interpolation retains gradients to the neighbor language branches.
    """
    results = []
    for bi in range(scene.xyz.shape[0]):
        language = scene.language[bi]
        output = language.clone()
        for instance in torch.unique(scene.instance_ids[bi][occluded[bi]]).tolist():
            if instance < 0:
                continue
            same = (scene.instance_ids[bi] == instance) & scene.valid[bi]
            targets = torch.where(same & occluded[bi])[0]
            donors = torch.where(same & ~occluded[bi])[0]
            if donors.numel() == 0:
                continue
            for part in targets.split(chunk_size):
                distances = torch.cdist(scene.xyz[bi,part].float(), scene.xyz[bi,donors].float())
                close = donors[distances.topk(min(neighbors,donors.numel()), largest=False).indices]
                mean = language[close].mean(1)
                output[part] = (1-strength)*language[part] + strength*mean
        results.append(output)
    return replace(scene, language=F.normalize(torch.stack(results), dim=-1))
