from __future__ import annotations

from dataclasses import dataclass, fields, replace
import torch


@dataclass
class GaussianScene:
    """Batched Gaussians in the CURRENT ego frame. Quaternion order is wxyz.

    language is full CLIP space (512); language_code optionally preserves the
    upstream LangSplat three-dimensional bottleneck for checkpoint compatibility.
    Unknown instance ids are -1, never treated as one shared background instance.
    """

    xyz: torch.Tensor                  # [B,N,3], meters
    log_scales: torch.Tensor           # [B,N,3]
    rotations: torch.Tensor            # [B,N,4], ego-frame wxyz
    opacity: torch.Tensor              # [B,N,1], activated [0,1]
    colors: torch.Tensor               # [B,N,3], activated [0,1]
    language: torch.Tensor             # [B,N,512]
    valid: torch.Tensor                # [B,N], bool
    instance_ids: torch.Tensor         # [B,N], long
    view_ids: torch.Tensor             # [B,N], long
    velocity: torch.Tensor | None = None
    dynamic_probability: torch.Tensor | None = None
    language_code: torch.Tensor | None = None

    def item(self, index: int) -> GaussianScene:
        return GaussianScene(**{f.name: None if getattr(self, f.name) is None else
                                getattr(self, f.name)[index:index + 1] for f in fields(self)})

    def to(self, device: torch.device | str) -> GaussianScene:
        return GaussianScene(**{f.name: None if getattr(self, f.name) is None else
                                getattr(self, f.name).to(device) for f in fields(self)})

    def at_time(self, seconds: float | torch.Tensor) -> GaussianScene:
        if self.velocity is None or self.dynamic_probability is None:
            return self
        return replace(self, xyz=self.xyz + self.velocity * self.dynamic_probability * seconds)

    def detached(self) -> GaussianScene:
        return GaussianScene(**{f.name: None if getattr(self,f.name) is None else
                                getattr(self,f.name).detach() for f in fields(self)})

    def packed(self, language_code: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.xyz, self.log_scales, self.rotations,
                          self.opacity, language_code], -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], -1)


def matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    """Stable branch choice for fixed camera extrinsics, including 180 deg turns."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    roots = torch.stack([1+m00+m11+m22, 1+m00-m11-m22,
                         1-m00+m11-m22, 1-m00-m11+m22], -1).clamp_min(0).sqrt()
    candidates = torch.stack([
        torch.stack([roots[..., 0]**2, m[..., 2, 1]-m[..., 1, 2],
                     m[..., 0, 2]-m[..., 2, 0], m[..., 1, 0]-m[..., 0, 1]], -1),
        torch.stack([m[..., 2, 1]-m[..., 1, 2], roots[..., 1]**2,
                     m[..., 1, 0]+m[..., 0, 1], m[..., 0, 2]+m[..., 2, 0]], -1),
        torch.stack([m[..., 0, 2]-m[..., 2, 0], m[..., 1, 0]+m[..., 0, 1],
                     roots[..., 2]**2, m[..., 2, 1]+m[..., 1, 2]], -1),
        torch.stack([m[..., 1, 0]-m[..., 0, 1], m[..., 0, 2]+m[..., 2, 0],
                     m[..., 2, 1]+m[..., 1, 2], roots[..., 3]**2], -1)], -2)
    candidates = candidates / (2 * roots.clamp_min(0.1)[..., :, None])
    index = roots.argmax(-1)
    return torch.nn.functional.normalize(candidates.gather(-2, index[..., None, None].expand(
        *index.shape, 1, 4)).squeeze(-2), dim=-1)
