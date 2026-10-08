from __future__ import annotations
import argparse
import json
from pathlib import Path
from .public_data import CAMERAS, json_rows, file_sha256


def main():
    parser = argparse.ArgumentParser(description="Build exact keyframe identity index from SDK metadata, without loading images or LiDAR")
    parser.add_argument("--nuscenes-root", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from nuscenes.utils.splits import create_splits_scenes
    directory = Path(args.nuscenes_root) / args.version
    tables = {name: {r["token"]: r for r in json_rows(directory / f"{name}.json")}
              for name in ("scene", "sample", "sensor", "calibrated_sensor")}
    splits = create_splits_scenes()
    train_names = set(splits["mini_train"] if args.version == "v1.0-mini" else splits["train"])
    val_names = set(splits["mini_val"] if args.version == "v1.0-mini" else splits["val"])
    frames = {}
    for scene in tables["scene"].values():
        split = "train" if scene["name"] in train_names else "val" if scene["name"] in val_names else "test"
        token, seen, position = scene["first_sample_token"], set(), 0
        while token:
            if token in seen: raise ValueError("Cycle in SDK sample chain")
            seen.add(token)
            sample = tables["sample"][token]
            if sample["scene_token"] != scene["token"]: raise ValueError("SDK scene chain mismatch")
            frames[token] = {"sample_token": token, "scene_token": scene["token"], "scene_name": scene["name"],
                             "frame_index": position, "timestamp": sample["timestamp"], "split": split,
                             "prev": sample["prev"], "next": sample["next"], "cameras": {}}
            token, position = sample["next"], position + 1
    for sd in json_rows(directory / "sample_data.json"):
        if not sd["is_key_frame"] or sd["sample_token"] not in frames: continue
        calib = tables["calibrated_sensor"][sd["calibrated_sensor_token"]]
        channel = tables["sensor"][calib["sensor_token"]]["channel"]
        metadata = {"sample_data_token": sd["token"], "filename": sd["filename"],
                    "timestamp": sd["timestamp"], "ego_pose_token": sd["ego_pose_token"],
                    "calibrated_sensor_token": sd["calibrated_sensor_token"]}
        if channel in CAMERAS:
            if channel in frames[sd["sample_token"]]["cameras"]:
                raise ValueError("Duplicate SDK camera keyframe")
            frames[sd["sample_token"]]["cameras"][channel] = metadata
        elif channel == "LIDAR_TOP":
            frames[sd["sample_token"]]["lidar"] = metadata
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {name: file_sha256(directory / f"{name}.json") for name in (*tables, "sample_data")}
    payload = {"format_version": 1, "meta": {"version": args.version, "metadata_sha256": hashes,
              "nuscenes_root": str(Path(args.nuscenes_root).resolve()), "camera_order": CAMERAS,
              "frame_index_definition": "zero-based chronological sample chain within scene; NOT author Gaussian frame index"},
              "frames": sorted(frames.values(), key=lambda x: (x["scene_name"], x["frame_index"]))}
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"frames": len(frames), "scenes": len(tables["scene"]), "output": str(output)}))


if __name__ == "__main__":
    main()
