from __future__ import annotations

from pathlib import Path
import copy
import torch
from torch import nn
import torch.nn.functional as F
from .scene import GaussianScene, matrix_to_quaternion, quaternion_multiply


def unproject(depth: torch.Tensor, intrinsics: torch.Tensor,
              camera_to_ego: torch.Tensor, image_size: tuple[int,int]) -> torch.Tensor:
    b,v,_,h,w = depth.shape
    ih,iw = image_size
    yy,xx = torch.meshgrid((torch.arange(h,device=depth.device)+0.5)*ih/h,
                          (torch.arange(w,device=depth.device)+0.5)*iw/w,indexing="ij")
    pixels = torch.stack([xx,yy,torch.ones_like(xx)],-1)
    rays = torch.einsum("bvij,hwj->bvhwi",torch.linalg.inv(intrinsics.float()),pixels.float())
    points = rays*depth.permute(0,1,3,4,2)
    return (torch.einsum("bvij,bvhwj->bvhwi",camera_to_ego[...,:3,:3].float(),points.float())
            + camera_to_ego[...,:3,3][:,:,None,None,:]).reshape(b,-1,3)


def resize_map(x: torch.Tensor, size: tuple[int,int], mode: str = "bilinear") -> torch.Tensor:
    b,v,c,h,w = x.shape
    kwargs = {"align_corners":False} if mode == "bilinear" else {}
    return F.interpolate(x.reshape(b*v,c,h,w),size=size,mode=mode,**kwargs).reshape(b,v,c,*size)


class OnlineGaussianReconstructor(nn.Module):
    """Official DrivingForward depth + Gaussian networks, with semantic/motion heads.

    Input: calibrated current six-view RGB, [0,1]. No per-scene optimization or
    prebuilt Gaussian files. Static/dynamic Gaussians share a probabilistic motion
    head; temporal supervision is provided by future rendering losses.
    """
    def __init__(self, config: dict):
        super().__init__()
        from .vendor.drivingforward.depth_network import DepthNetwork
        from .vendor.drivingforward.gaussian_network import GaussianNetwork
        self.config = copy.deepcopy(config)
        cfg = config["drivingforward"]
        cfg = copy.deepcopy(cfg)
        # The upstream fusion network stores B-dependent grids. Rebuilt per batch
        # below only changes grids; learned weights remain the same.
        self.depth_net = DepthNetwork(cfg)
        self.gaussian_net = GaussianNetwork()
        self.stride = int(config.get("gaussian_stride",8))
        self.language_head = nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.SiLU(),nn.Conv2d(128,512,1))
        self.motion_head = nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,4,1))
        self.color_head = nn.Conv2d(64,3,1)
        self.scale_residual = nn.Conv2d(64,3,1)
        self.offset_head = nn.Conv2d(64,3,1)
        nn.init.zeros_(self.color_head.weight); nn.init.zeros_(self.color_head.bias)
        nn.init.zeros_(self.offset_head.weight); nn.init.zeros_(self.offset_head.bias)
        nn.init.constant_(self.motion_head[-1].bias,0)
        self.max_speed = float(config.get("max_speed",30))
        self.min_depth = float(cfg["training"]["min_depth"])
        self.max_depth = float(cfg["training"]["max_depth"])
        self.focal_scale = float(cfg["training"]["focal_length_scale"])
        weights = config.get("weights_dir")
        if weights:
            for name,module in [("depth_net",self.depth_net),("gs_net",self.gaussian_net)]:
                path = Path(weights)/f"{name}.pth"
                state = torch.load(path,map_location="cpu",weights_only=True)
                current = module.state_dict()
                # Original checkpoints include scalar height/width metadata.
                state = {k.removeprefix("module."):v for k,v in state.items()
                         if k.removeprefix("module.") in current}
                module.load_state_dict(state,strict=True)

    def _inputs(self, batch: dict) -> dict:
        image,k,c2e = batch["images"],batch["intrinsics"],batch["camera_to_ego"]
        b,v,_,h,w = image.shape
        expected = self.config["drivingforward"]["training"]
        if (h,w) != (expected["height"],expected["width"]) or v != 6:
            raise ValueError("DrivingForward expects six views at configured height/width")
        fusion = self.depth_net.fusion_net
        if b != self.depth_net.batch_size:
            raise ValueError("DrivingForward batch size must match reconstruction config; use per_device_batch_size=1")
        mask = batch.get("image_mask",torch.ones(b,v,1,h,w,device=image.device))
        inputs = {("color_aug",0,0):image, "extrinsics":c2e,
                  "extrinsics_inv":torch.linalg.inv(c2e.float()), "mask":mask}
        for level in range(4):
            kl = torch.eye(4,device=k.device).expand(b,v,4,4).clone()
            kl[...,:3,:3] = k.float()
            kl[...,:2,:] /= 2**level
            inputs[("K",level)] = kl
            inputs[("inv_K",level)] = torch.linalg.inv(kl)
        return inputs

    def forward(self, batch: dict) -> GaussianScene:
        images = batch["images"]
        b,v,_,h,w = images.shape
        output = self.depth_net(self._inputs(batch))
        target = (h//self.stride,w//self.stride)
        depths,rotations,scales,opacity,features = [],[],[],[],[]
        for vi in range(v):
            item = output[("cam",vi)]
            disp = F.interpolate(item[("disp",0)],(h,w),mode="bilinear",align_corners=False)
            depth = (1/(1/self.max_depth+(1/self.min_depth-1/self.max_depth)*disp))
            depth = depth*batch["intrinsics"][:,vi,0,0,None,None,None]/self.focal_scale
            feat = item[("img_feat",0,0)]
            rot,scale,op,_sh = self.gaussian_net(images[:,vi],depth,feat)
            depths.append(F.interpolate(depth,target,mode="bilinear",align_corners=False))
            rotations.append(F.interpolate(rot,target,mode="bilinear",align_corners=False))
            scales.append(F.interpolate(scale,target,mode="bilinear",align_corners=False))
            opacity.append(F.interpolate(op,target,mode="bilinear",align_corners=False))
            features.append(F.interpolate(feat[0],target,mode="bilinear",align_corners=False))
        feature = torch.stack(features,1).reshape(b*v,64,*target)
        def flatten(x):
            return x.reshape(b,v,x.shape[1],*target).permute(0,1,3,4,2).reshape(b,-1,x.shape[1])
        lang = F.normalize(flatten(self.language_head(feature)),dim=-1)
        motion = flatten(self.motion_head(feature))
        xyz = unproject(torch.stack(depths,1),batch["intrinsics"],batch["camera_to_ego"],(h,w))
        local_offset = torch.tanh(flatten(self.offset_head(feature))) * 0.5
        xyz = xyz + local_offset
        rot_cam = torch.stack(rotations,1).permute(0,1,3,4,2)
        rot_ego = matrix_to_quaternion(batch["camera_to_ego"][...,:3,:3]).float()[:,:,None,None,:]
        rotation = F.normalize(quaternion_multiply(rot_ego,rot_cam.float()).reshape(b,-1,4),dim=-1)
        # Keep official scales, add learnable stride compensation for subsampling.
        scale = torch.stack(scales,1).permute(0,1,3,4,2).reshape(b,-1,3)
        scale = scale*self.stride*F.softplus(flatten(self.scale_residual(feature))).clamp_min(0.1)
        rgb = resize_map(images,target).permute(0,1,3,4,2).reshape(b,-1,3)
        rgb = (rgb + 0.1*torch.tanh(flatten(self.color_head(feature)))).clamp(0,1)
        op = torch.stack(opacity,1).permute(0,1,3,4,2).reshape(b,-1,1)
        valid = torch.isfinite(xyz).all(-1)
        instance = batch.get("instance_ids")
        if instance is None:
            ids = torch.full(valid.shape,-1,dtype=torch.long,device=xyz.device)
        else:
            ids = resize_map(instance.float().unsqueeze(2),target,"nearest").reshape(b,-1).long()
        views = torch.arange(v,device=xyz.device)[None,:,None].expand(b,v,target[0]*target[1]).reshape(b,-1)
        return GaussianScene(xyz,scale.clamp_min(1e-5).log(),rotation,op,rgb,lang,valid,ids,views,
                             torch.tanh(motion[...,:3])*self.max_speed,torch.sigmoid(motion[...,3:4]))
