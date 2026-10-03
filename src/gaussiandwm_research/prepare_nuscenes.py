from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


CAMERAS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK_LEFT","CAM_BACK_RIGHT","CAM_BACK"]


def pose(record):
    from pyquaternion import Quaternion
    transform = np.eye(4,dtype=np.float64)
    transform[:3,:3] = Quaternion(record["rotation"]).rotation_matrix
    transform[:3,3] = record["translation"]
    return transform


def ego_pose(nusc,sample):
    sd = nusc.get("sample_data",sample["data"]["CAM_FRONT"])
    return pose(nusc.get("ego_pose",sd["ego_pose_token"]))


def camera_metadata(nusc,sample,current_ego_to_world):
    paths,ks,transforms = [],[],[]
    for camera in CAMERAS:
        sd = nusc.get("sample_data",sample["data"][camera])
        calibrated = nusc.get("calibrated_sensor",sd["calibrated_sensor_token"])
        camera_to_world = pose(nusc.get("ego_pose",sd["ego_pose_token"])) @ pose(calibrated)
        paths.append(sd["filename"])
        ks.append(calibrated["camera_intrinsic"])
        transforms.append((np.linalg.inv(current_ego_to_world) @ camera_to_world).tolist())
    return paths,ks,transforms


def lidar_depth(nusc,sample,paths,ks,camera_to_current,current_to_world,
                cache_root: Path,height: int,width: int):
    from nuscenes.utils.data_classes import LidarPointCloud
    from PIL import Image
    sd = nusc.get("sample_data",sample["data"]["LIDAR_TOP"])
    calib = nusc.get("calibrated_sensor",sd["calibrated_sensor_token"])
    cloud = LidarPointCloud.from_file(str(Path(nusc.dataroot)/sd["filename"]))
    lidar_to_world = pose(nusc.get("ego_pose",sd["ego_pose_token"])) @ pose(calib)
    homogeneous = np.vstack([cloud.points[:3],np.ones((1,cloud.points.shape[1]))])
    current_xyz = np.linalg.inv(current_to_world) @ lidar_to_world @ homogeneous
    result = []
    for vi,path in enumerate(paths):
        cam_xyz = np.linalg.inv(np.asarray(camera_to_current[vi])) @ current_xyz
        with Image.open(Path(nusc.dataroot)/path) as image:
            ih,iw = image.height,image.width
        k = np.asarray(ks[vi],dtype=float).copy()
        k[0] *= width/iw; k[1] *= height/ih
        pixels = k @ cam_xyz[:3]
        xy = np.rint(pixels[:2]/np.maximum(pixels[2:],1e-6)).astype(np.int64)
        valid = (cam_xyz[2]>0.1)&(xy[0]>=0)&(xy[0]<width)&(xy[1]>=0)&(xy[1]<height)
        depth = np.full(height*width,np.inf,dtype=np.float32)
        np.minimum.at(depth,xy[1,valid]*width+xy[0,valid],cam_xyz[2,valid].astype(np.float32))
        depth[~np.isfinite(depth)] = 0
        filename = cache_root/f"{sample['token']}_{CAMERAS[vi]}.npz"
        filename.parent.mkdir(parents=True,exist_ok=True)
        if not filename.exists():
            np.savez_compressed(filename,depth=depth.reshape(height,width))
        result.append(str(filename.resolve()))
    return result


def scene_supervision(nusc,sample,current_to_world):
    """TRAINING-ONLY coarse JSON supervision from real nuScenes box annotations."""
    from pyquaternion import Quaternion
    elements = []
    counts = {}
    transform = np.linalg.inv(current_to_world)
    for token in sample["anns"]:
        annotation = nusc.get("sample_annotation",token)
        category = annotation["category_name"]
        counts[category] = counts.get(category,0)+1
        box = nusc.get_box(token)
        corners = transform @ np.vstack([box.corners(),np.ones((1,8))])
        bounds = [*corners[:3].min(1).tolist(),*corners[:3].max(1).tolist()]
        distance = np.linalg.norm((np.asarray(bounds[:3])+np.asarray(bounds[3:]))/2)
        elements.append({"name":category,"weight":1/(1+distance/30),"bounds":bounds,
                         "annotation_token":token})
    elements.sort(key=lambda x:x["weight"],reverse=True)
    return {"summary":", ".join(f"{count} {category}" for category,count in sorted(counts.items())),
            "complexity":min(1,len(elements)/32),"elements":elements[:16],"target_bounds":None,
            "source":"nuScenes_GT_boxes_training_only"}


def main():
    parser = argparse.ArgumentParser(description="Join real QA annotations with calibrated nuScenes current/future frames")
    parser.add_argument("--nuscenes-root",required=True)
    parser.add_argument("--version",default="v1.0-trainval")
    parser.add_argument("--split",choices=["train","val","mini_train","mini_val"],default="train")
    parser.add_argument("--qa-jsonl",required=True,help="Rows: sample_token, query, answer; optional task_kind, scene_hint")
    parser.add_argument("--output",required=True)
    parser.add_argument("--depth-cache",required=True)
    parser.add_argument("--horizon",type=int,default=6)
    parser.add_argument("--image-height",type=int,default=352)
    parser.add_argument("--image-width",type=int,default=640)
    args = parser.parse_args()
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    nusc = NuScenes(version=args.version,dataroot=args.nuscenes_root,verbose=True)
    split = set(create_splits_scenes()[args.split])
    root = Path(args.depth_cache)
    output = Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    counts = {"written":0,"wrong_split":0,"missing_future":0}
    with output.open("w",encoding="utf-8") as handle:
        for line in Path(args.qa_jsonl).read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            qa = json.loads(line)
            sample = nusc.get("sample",qa["sample_token"])
            if nusc.get("scene",sample["scene_token"])["name"] not in split:
                counts["wrong_split"] += 1; continue
            future = []
            cursor = sample
            for _ in range(args.horizon):
                if not cursor["next"]: break
                cursor = nusc.get("sample",cursor["next"]); future.append(cursor)
            if len(future) != args.horizon:
                counts["missing_future"] += 1; continue
            current_to_world = ego_pose(nusc,sample)
            images,ks,cameras = camera_metadata(nusc,sample,current_to_world)
            depth = lidar_depth(nusc,sample,images,ks,cameras,current_to_world,root,args.image_height,args.image_width)
            item = {"sample_uid":qa.get("sample_uid",sample["token"]+f"_{counts['written']}"),
                    "sample_token":sample["token"],"scene_token":sample["scene_token"],
                    "query":qa["query"],"answer":qa["answer"],"task_kind":qa.get("task_kind","global"),
                    "scene_hint":qa.get("scene_hint") or scene_supervision(nusc,sample,current_to_world),"image_paths":images,"intrinsics":ks,
                    "camera_to_ego":cameras,"depth_paths":depth,
                    "future_image_paths":[],"future_depth_paths":[],"future_camera_to_ego":[],"trajectory":[]}
            for frame in future:
                paths,future_k,poses = camera_metadata(nusc,frame,current_to_world)
                item["future_image_paths"].append(paths)
                item["future_camera_to_ego"].append(poses)
                item["future_depth_paths"].append(lidar_depth(nusc,frame,paths,future_k,poses,current_to_world,
                    root,args.image_height,args.image_width))
                relative = np.linalg.inv(current_to_world) @ ego_pose(nusc,frame)
                yaw = np.arctan2(relative[1,0],relative[0,0])
                item["trajectory"].append([*relative[:3,3].tolist(),float(np.sin(yaw)),float(np.cos(yaw))])
            # Command labels are an explicit derived supervision convention.
            end = item["trajectory"][-1]
            heading = np.arctan2(end[3],end[4])
            item["command"] = "brake" if np.linalg.norm(end[:2])<0.5 else "left" if heading>0.15 else "right" if heading<-.15 else "straight"
            item["command_label_source"] = "trajectory_derived_thresholds_0.5m_0.15rad"
            # Preserve explicitly provided offline Gaussians and feature metadata.
            for key in ["gauss_paths","gauss_to_ego","gauss_instance_ids_path","clip_text_feature_path",
                        "semantic_feature_paths","instance_id_paths"]:
                if key in qa: item[key] = qa[key]
            handle.write(json.dumps(item,ensure_ascii=False)+"\n")
            counts["written"] += 1
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
