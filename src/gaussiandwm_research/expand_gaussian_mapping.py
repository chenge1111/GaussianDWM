from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Expand verified author scene/frame mapping onto Gaussian inventory paths")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--frame-map", required=True, help="JSONL: author_scene_key, author_frame_key, optional view, sample_token, evidence, gauss_to_ego/coordinate_frame")
    parser.add_argument("--output", required=True)
    parser.add_argument("--unmatched-output", required=True)
    args = parser.parse_args()
    mapping = {}
    with Path(args.frame_map).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line)
            if not row.get("evidence") or not row.get("sample_token"):
                raise ValueError("Verified mapping needs sample_token and evidence")
            if "gauss_to_ego" not in row and "coordinate_frame" not in row:
                raise ValueError("Verified mapping needs a transform or known coordinate frame")
            key = (str(row["author_scene_key"]), str(row["author_frame_key"]), row.get("view"))
            if key in mapping and mapping[key] != row: raise ValueError("Conflicting author scene/frame mapping")
            mapping[key] = row
    output, unmatched = Path(args.output), Path(args.unmatched_output)
    output.parent.mkdir(parents=True, exist_ok=True); unmatched.parent.mkdir(parents=True, exist_ok=True)
    counts = {"matched": 0, "unmatched": 0}
    with output.open("w", encoding="utf-8") as target, unmatched.open("w", encoding="utf-8") as rejects, Path(args.inventory).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip(): continue
            item = json.loads(line)
            prefix = (str(item["author_scene_key"]), str(item["author_frame_key"]))
            row = mapping.get((*prefix, item["view_from_path"]), mapping.get((*prefix, None)))
            if row is None:
                rejects.write(json.dumps(item, ensure_ascii=False) + "\n"); counts["unmatched"] += 1
            else:
                result = {"path": item["path"], "sample_token": row["sample_token"], "view": item["view_from_path"],
                          "evidence": row["evidence"]}
                if "gauss_to_ego" in row: result["gauss_to_ego"] = row["gauss_to_ego"]
                else: result["coordinate_frame"] = row["coordinate_frame"]
                target.write(json.dumps(result, ensure_ascii=False) + "\n"); counts["matched"] += 1
    print(json.dumps(counts))
    if not counts["matched"]: raise SystemExit("No verified author-frame matches; no order-based fallback is permitted")


if __name__ == "__main__":
    main()
