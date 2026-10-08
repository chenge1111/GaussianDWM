from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def read_array(path: Path,key: str | None = None):
    if path.suffix == ".npz":
        with np.load(path,allow_pickle=False) as archive:
            return np.asarray(archive[key or archive.files[0]])
    if path.suffix == ".npy":
        return np.load(path,allow_pickle=False)
    if path.suffix in {".pt",".pth"}:
        value = torch.load(path,map_location="cpu",weights_only=True)
        return value[key] if isinstance(value,dict) else value
    return np.asarray(Image.open(path))


class DrivingManifestDataset(Dataset):
    """Explicit JSONL data contract, shared by offline and online experiments.

    All future poses and trajectory labels use the CURRENT ego frame. Calibrations
    refer to unresized images; loader scales intrinsics when resizing. Required
    supervision is enforced by the selected training mode, not fabricated here.
    """
    def __init__(self,manifest: str,root: str,image_height: int,image_width: int,
                 mode: str = "online",require_answer: bool = True):
        self.records = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.records:
            raise ValueError("Manifest is empty")
        self.root,self.size,self.mode = Path(root),(image_height,image_width),mode
        self.require_answer = require_answer

    def __len__(self):
        return len(self.records)

    def path(self,value):
        path = Path(value)
        return path if path.is_absolute() else self.root/path

    def images(self,paths):
        h,w = self.size
        images,sizes = [],[]
        for path in paths:
            with Image.open(self.path(path)) as image:
                image = image.convert("RGB")
                sizes.append((image.height,image.width))
                images.append(torch.from_numpy(np.array(image.resize((w,h)),copy=True)).permute(2,0,1).float()/255)
        return torch.stack(images),sizes

    def maps(self,paths,channels: int = 1,nearest: bool = True):
        arrays = []
        for path in paths:
            x = torch.as_tensor(read_array(self.path(path))).float()
            if x.ndim == 2:
                x = x[None]
            if x.ndim != 3 or x.shape[0] != channels:
                raise ValueError(f"Expected [{channels},H,W] map at {path}, got {tuple(x.shape)}")
            if channels == 512:
                # Keep teacher features at their native (lower) resolution.
                arrays.append(x); continue
            x = torch.nn.functional.interpolate(x[None],self.size,mode="nearest" if nearest else "bilinear",
                                                **({} if nearest else {"align_corners":False}))[0]
            arrays.append(x)
        return torch.stack(arrays)

    def __getitem__(self,index):
        record = self.records[index]
        if len(record["image_paths"]) != 6:
            raise ValueError("Manifest must provide six surround-view image paths")
        images,sizes = self.images(record["image_paths"])
        k = torch.tensor(record["intrinsics"],dtype=torch.float32)
        h,w = self.size
        for vi,(ih,iw) in enumerate(sizes):
            k[vi,0,:] *= w/iw; k[vi,1,:] *= h/ih
        sample = {"sample_uid":record["sample_uid"],"query":record["query"],"images":images,
                  "intrinsics":k,"camera_to_ego":torch.tensor(record["camera_to_ego"],dtype=torch.float32),
                  "task_kind":record.get("task_kind","global"),"scene_hint":record.get("scene_hint",{})}
        if self.require_answer and not record.get("answer"):
            raise ValueError("QA training requires nonempty ground-truth answer")
        sample["answer"] = record.get("answer","")
        feature = torch.as_tensor(read_array(self.path(record["clip_text_feature_path"]))).float().reshape(-1)
        if feature.numel() != 512:
            raise ValueError("CLIP text feature must contain 512 values")
        sample["clip_text_feature"] = feature
        if "depth_paths" in record:
            sample["depth"] = self.maps(record["depth_paths"])
        if "semantic_feature_paths" in record:
            sample["semantic_features"] = self.maps(record["semantic_feature_paths"],512)
        if "instance_id_paths" in record:
            sample["instance_ids"] = self.maps(record["instance_id_paths"]).squeeze(1).long()
        if "future_image_paths" in record:
            sample["future_images"] = torch.stack([self.images(paths)[0] for paths in record["future_image_paths"]])
        if "future_depth_paths" in record:
            sample["future_depth"] = torch.stack([self.maps(paths) for paths in record["future_depth_paths"]])
        if "future_camera_to_ego" in record:
            sample["future_camera_to_ego"] = torch.tensor(record["future_camera_to_ego"],dtype=torch.float32)
        if "trajectory" in record:
            sample["trajectory"] = torch.tensor(record["trajectory"],dtype=torch.float32)
        if "command" in record:
            value = record["command"]
            sample["command"] = torch.tensor(["brake","straight","left","right"].index(value) if isinstance(value,str) else value)
        if self.mode == "offline":
            from gaussiandwm_cvpr.data.gauss_normalizer import GaussNormalizer
            normalizer = GaussNormalizer()
            values,view_ids = [],[]
            for vi,path in enumerate(record["gauss_paths"]):
                data = normalizer.load_and_normalize([str(self.path(path))],scene_idx=None,frame_idx=None)
                if "gauss_to_ego" in record:
                    from .scene import matrix_to_quaternion,quaternion_multiply
                    transform = torch.tensor(record["gauss_to_ego"][vi],dtype=torch.float32)
                    xyz = data[:,:3] @ transform[:3,:3].T+transform[:3,3]
                    quaternion = matrix_to_quaternion(transform[:3,:3])[None]
                    data = torch.cat([xyz,data[:,3:6],quaternion_multiply(quaternion,data[:,6:10]),data[:,10:]],-1)
                from .public_data import CAMERAS
                view_name = record.get("gauss_view_names",[])
                view_id = CAMERAS.index(view_name[vi]) if view_name else vi
                values.append(data); view_ids.append(torch.full((len(data),),view_id,dtype=torch.long))
            sample["gauss_values"] = torch.cat(values)
            sample["gauss_view_ids"] = torch.cat(view_ids)
            if "gauss_instance_ids_path" in record:
                sample["gauss_instance_ids"] = torch.as_tensor(read_array(self.path(record["gauss_instance_ids_path"]))).long()
        return sample


def single_scene_collate(samples):
    if len(samples) != 1:
        raise ValueError("Variable-budget training requires per-device batch=1")
    return {key:value[None] if isinstance(value,torch.Tensor) else value for key,value in samples[0].items()}
