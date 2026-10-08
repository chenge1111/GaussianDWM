from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from .public_data import FrameIndex, CAMERAS


def main():
    parser = argparse.ArgumentParser(description="Data admission report for public-data matching, performed on the data host")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mode", choices=["offline", "online"], required=True)
    parser.add_argument("--require-features", action="store_true")
    parser.add_argument("--require-world", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    index, root = FrameIndex(args.index), Path(args.data_root)
    counts, scenes, samples, uids = Counter(), defaultdict(set), defaultdict(set), set()
    directory = Path(args.output_dir); directory.mkdir(parents=True, exist_ok=True)
    def resolve(value):
        path = Path(value); return path if path.is_absolute() else root / path
    def exists(value): return resolve(value).is_file()
    with (directory / "rejected.jsonl").open("w", encoding="utf-8") as reject:
        for filename in args.manifest:
            with Path(filename).open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip(): continue
                    counts["input"] += 1
                    try:
                        row = json.loads(line); frame = index.samples[row["sample_token"]]
                        split = frame["split"]
                        if row["scene_token"] != frame["scene_token"] or row.get("split", split) != split:
                            raise ValueError("SDK sample/scene/split conflict")
                        if row["sample_uid"] in uids: raise ValueError("Duplicate sample_uid across manifests")
                        uids.add(row["sample_uid"])
                        expected = [frame["cameras"][v]["filename"] for v in CAMERAS]
                        if row["image_paths"] != expected: raise ValueError("Camera order/path does not match SDK keyframe")
                        if not all(exists(path) for path in expected): raise ValueError("Missing RGB images")
                        if not row.get("query") or not row.get("answer"): raise ValueError("Missing genuine QA labels")
                        if "<embeding>" in row["answer"] or "<embedding>" in row["answer"]:
                            raise ValueError("Unresolved numeric grounding placeholder")
                        if "intrinsics" not in row or "camera_to_ego" not in row:
                            raise ValueError("Prepare calibrated manifest before admission")
                        if args.require_features:
                            if not exists(row["clip_text_feature_path"]): raise ValueError("Missing CLIP query feature")
                            import numpy as np
                            feature = np.load(resolve(row["clip_text_feature_path"]), allow_pickle=False)
                            if feature.shape != (512,) or not np.isfinite(feature).all(): raise ValueError("Invalid CLIP text feature")
                        if args.mode == "offline":
                            if not row.get("gauss_paths") or not all(exists(path) for path in row["gauss_paths"]):
                                raise ValueError("Missing confirmed Gaussian files")
                            if len(row["gauss_paths"]) != len(row.get("gauss_to_ego", [])):
                                raise ValueError("Missing per-Gaussian coordinate transforms")
                        if args.require_world:
                            if not row.get("future_depth_paths") or not row.get("future_image_paths"):
                                raise ValueError("World training needs true future images and depth")
                            horizon = len(row["future_image_paths"])
                            if horizon != len(row["future_depth_paths"]) or horizon != len(row.get("trajectory", [])):
                                raise ValueError("Future targets/trajectory horizon mismatch")
                            if len(row.get("future_camera_to_ego", [])) != horizon:
                                raise ValueError("Missing future camera transforms")
                            for values in (*row["future_image_paths"], *row["future_depth_paths"]):
                                if len(values) != 6 or not all(exists(path) for path in values): raise ValueError("Missing six-view future targets")
                            for token in row.get("future_sample_tokens", []):
                                if index.samples[token]["scene_token"] != frame["scene_token"]:
                                    raise ValueError("Future frame crosses scene boundary")
                        counts[f"admitted:{split}"] += 1
                        scenes[split].add(frame["scene_token"]); samples[split].add(frame["sample_token"])
                    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
                        counts["rejected:" + str(exc)] += 1
                        reject.write(json.dumps({"input_file": filename, "line": line_number, "reason": str(exc)}) + "\n")
    overlap = sorted(scenes["train"] & scenes["val"])
    if overlap: counts["scene_leakage"] = len(overlap)
    summary = {"counts": dict(counts), "sample_counts": {k: len(v) for k, v in samples.items()},
               "scenes": {k: sorted(v) for k, v in scenes.items()}, "train_val_scene_overlap": overlap,
               "mode": args.mode, "sdk_meta": index.meta, "require_world": args.require_world,
               "require_features": args.require_features,
               "protocol": "identity/path/data-admission checks; no model-quality or paper-metric claim"}
    (directory / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"counts": summary["counts"], "sample_counts": summary["sample_counts"]}))
    if overlap or any(key.startswith("rejected:") for key in counts):
        raise SystemExit("Manifest admission failed; inspect report and rejection rows")


if __name__ == "__main__":
    main()
