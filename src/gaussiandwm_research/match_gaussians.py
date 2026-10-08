from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from .public_data import FrameIndex, CAMERAS, TOKEN, Report, path_key


def inferred_identity(path, index, basename_lookup):
    """Recognize literal SDK identities, never author scene/frame ordinal guesses."""
    evidence = []
    token_name = path.stem
    if token_name in index.samples:
        evidence.append(("sample_token_filename", token_name))
    if token_name in index.sd_to_sample:
        evidence.append(("sample_data_token_filename", index.sd_to_sample[token_name]))
    # Accept an exact unique original camera filename retained in the Gaussian
    # filename (e.g. n015-...__CAM_FRONT__timestamp.jpg.pth).
    for name in (token_name, token_name + ".jpg", token_name + ".png"):
        tokens = basename_lookup.get(name, set())
        if len(tokens) == 1:
            evidence.append(("exact_unique_SDK_camera_basename", next(iter(tokens))))
    return evidence


def matrix_for(coordinate_frame, frame, view, sdk_root, cache):
    if coordinate_frame == "current_ego":
        return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    if coordinate_frame not in {"camera", "world"}:
        raise ValueError("Gaussian coordinate frame unconfirmed; provide gauss_to_ego or explicit coordinate_frame")
    from .prepare_nuscenes import pose
    if not cache:
        for table in ("ego_pose", "calibrated_sensor"):
            cache[table] = {row["token"]: row for row in json.loads((sdk_root / f"{table}.json").read_text(encoding="utf-8"))}
    import numpy as np
    ref = frame["cameras"]["CAM_FRONT"]
    current_to_world = pose(cache["ego_pose"][ref["ego_pose_token"]])
    world_to_current = np.linalg.inv(current_to_world)
    if coordinate_frame == "world": return world_to_current.tolist()
    camera = frame["cameras"][view]
    camera_to_world = pose(cache["ego_pose"][camera["ego_pose_token"]]) @ pose(cache["calibrated_sensor"][camera["calibrated_sensor_token"]])
    return (world_to_current @ camera_to_world).tolist()


def valid_transform(value):
    import numpy as np
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4,4) or not np.isfinite(matrix).all(): raise ValueError("gauss_to_ego must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0,0,0,1], atol=1e-5): raise ValueError("Invalid homogeneous transform bottom row")
    rotation = matrix[:3,:3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3):
        raise ValueError("Gaussian transform must be rigid; scale/unit changes require an explicit data conversion")
    return matrix.tolist()


def main():
    parser = argparse.ArgumentParser(description="Inventory Gaussian files and join only exact/explicit SDK frame identities")
    parser.add_argument("--gauss-root", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mapping", help="JSONL per file: path, sample_token, view, gauss_to_ego/coordinate_frame, evidence")
    parser.add_argument("--coordinate-frame", choices=["current_ego", "camera", "world"],
                        help="Use only after verifying a uniform convention for all files")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--min-views", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.min_views <= 6: parser.error("min-views must be between 1 and 6")
    index = FrameIndex(args.index)
    root = Path(args.gauss_root).resolve()
    sdk_root = Path(index.meta["nuscenes_root"]) / index.meta["version"]
    overrides, conflicts = {}, set()
    if args.mapping:
        for line in Path(args.mapping).read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            path = Path(row["path"])
            key = str((path if path.is_absolute() else root / path).resolve())
            if key in overrides and overrides[key] != row: conflicts.add(key)
            overrides[key] = row
    basenames = defaultdict(set)
    for key, token in index.image_to_sample.items(): basenames[Path(key).name].add(token)
    report, grouped, cache, observed = Report(args.output_dir), defaultdict(dict), {}, set()
    with (Path(args.output_dir) / "inventory.jsonl").open("w", encoding="utf-8") as inventory:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".pth", ".pt", ".npz", ".npy"}: continue
            observed.add(str(path.resolve()))
            relative = path.relative_to(root).as_posix()
            view_match = re.search(r"CAM_(?:FRONT_LEFT|FRONT_RIGHT|BACK_LEFT|BACK_RIGHT|FRONT|BACK)(?![A-Z_])", relative)
            author = re.search(r"(?:^|/)([^/]+)_CAM_", relative)
            item = {"path": relative, "bytes": path.stat().st_size,
                    "view_from_path": view_match.group(0) if view_match else None,
                    "author_scene_key": author.group(1) if author else None,
                    "author_frame_key": path.stem,
                    "identity_candidates": inferred_identity(path, index, basenames)}
            inventory.write(json.dumps(item, ensure_ascii=False) + "\n")
            report.counts["gaussian_files"] += 1
            if args.inventory_only: continue
            try:
                key = str(path.resolve())
                if key in conflicts: raise ValueError("Conflicting explicit mappings for same Gaussian file")
                row = overrides.get(key, {})
                evidence = list(item["identity_candidates"])
                if row:
                    if not row.get("evidence"): raise ValueError("Explicit mapping needs recorded identity evidence")
                    token = row.get("sample_token")
                    if token not in index.samples: raise ValueError("Explicit sample token is not in this SDK index")
                    evidence.append(("explicit_mapping:" + str(row["evidence"]), token))
                tokens = {token for _, token in evidence}
                if len(tokens) != 1:
                    raise ValueError("No exact identity; numeric scene/frame names require explicit author mapping" if not tokens else "Conflicting Gaussian identity evidence")
                token = next(iter(tokens)); frame = index.samples[token]
                view = row.get("view", item["view_from_path"])
                if view not in CAMERAS: raise ValueError("Gaussian view is missing or invalid")
                if item["view_from_path"] and view != item["view_from_path"]: raise ValueError("Explicit view disagrees with filename")
                if "gauss_to_ego" in row:
                    transform = valid_transform(row["gauss_to_ego"])
                    convention = "explicit_rigid_transform"
                else:
                    convention = row.get("coordinate_frame", args.coordinate_frame)
                    transform = matrix_for(convention, frame, view, sdk_root, cache)
                if view in grouped[token]:
                    grouped[token][view] = None
                    raise ValueError("Multiple Gaussian files for same frame/view; select one reconstruction variant")
                grouped[token][view] = {"path": str(path.resolve()), "gauss_to_ego": transform,
                                        "evidence": evidence, "coordinate_frame": convention}
                report.counts["identity_and_coordinates_matched_files"] += 1
            except (ValueError, KeyError, TypeError) as exc:
                report.reject(str(exc), path=relative)
    for key in overrides.keys() - observed:
        report.reject("explicit_mapping_file_missing", path=key)
    with (Path(args.output_dir) / "frames.gaussians.jsonl").open("w", encoding="utf-8") as out:
        if not args.inventory_only:
            for token, views in sorted(grouped.items()):
                views = {v: item for v, item in views.items() if item is not None}
                if len(views) < args.min_views:
                    report.reject("insufficient_confirmed_views", sample_token=token, views=sorted(views)); continue
                names = [v for v in CAMERAS if v in views]
                frame = index.samples[token]
                row = {"sample_token": token, "scene_token": frame["scene_token"], "scene_name": frame["scene_name"],
                       "split": frame["split"], "gauss_paths": [views[v]["path"] for v in names],
                       "gauss_view_names": names, "gauss_to_ego": [views[v]["gauss_to_ego"] for v in names],
                       "gaussian_match_evidence": {v: views[v]["evidence"] for v in names}}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                report.accepted(dict(row, qa_group="gaussian_frame"))
    result = report.finish(index_meta=index.meta, inventory_only=args.inventory_only,
                           min_views=args.min_views, author_scene_number_matching="disabled")
    print(json.dumps(result["counts"]))
    if not args.inventory_only and not any(key.startswith("written:") for key in report.counts):
        raise SystemExit("No confirmed Gaussian frames; use inventory and obtain explicit frame/coordinate mapping")


if __name__ == "__main__":
    main()
