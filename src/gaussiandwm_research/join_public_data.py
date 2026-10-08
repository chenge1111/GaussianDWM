from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from .public_data import FrameIndex, Report, CAMERAS


def main():
    parser = argparse.ArgumentParser(description="Join converted QA with confirmed Gaussian maps or prepare online QA")
    parser.add_argument("--qa-jsonl", nargs="+", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--mode", choices=["offline", "online"], required=True)
    parser.add_argument("--gaussian-map", help="frames.gaussians.jsonl, required for offline")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    if args.mode == "offline" and not args.gaussian_map: parser.error("Offline matching needs --gaussian-map")
    index = FrameIndex(args.index)
    gaussians = {}
    if args.gaussian_map:
        for line in Path(args.gaussian_map).read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            if row["sample_token"] in gaussians: raise ValueError("Duplicate frame in Gaussian map")
            gaussians[row["sample_token"]] = row
    report, seen_uids, content_hashes = Report(args.output_dir), set(), set()
    with ExitStack() as stack:
        outputs = {split: stack.enter_context((Path(args.output_dir) / f"{split}.matched.qa.jsonl").open("w", encoding="utf-8")) for split in ("train", "val")}
        for filename in args.qa_jsonl:
            with Path(filename).open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip(): continue
                    report.counts["input_qa"] += 1
                    try:
                        row = json.loads(line)
                        frame = index.samples[row["sample_token"]]
                        if row["scene_token"] != frame["scene_token"] or row["split"] != frame["split"]:
                            raise ValueError("QA identity/split disagrees with SDK index")
                        if frame["split"] not in outputs: raise ValueError("Non-trainval row")
                        if row["sample_uid"] in seen_uids:
                            report.counts["duplicate_sample_uid"] += 1; continue
                        fingerprint = (row["sample_token"], row["query"], row["answer"])
                        if fingerprint in content_hashes:
                            report.counts["duplicate_QA_content"] += 1; continue
                        row["image_paths"] = [frame["cameras"][v]["filename"] for v in CAMERAS]
                        if args.check_images:
                            missing = [image for image in row["image_paths"] if not (Path(index.meta["nuscenes_root"]) / image).is_file()]
                            if missing: raise ValueError("Missing current RGB image files")
                        if args.mode == "offline":
                            gaussian = gaussians.get(row["sample_token"])
                            if gaussian is None: raise ValueError("QA has no confirmed Gaussian frame")
                            if gaussian["scene_token"] != frame["scene_token"] or gaussian["split"] != frame["split"]:
                                raise ValueError("Gaussian map scene/split conflict")
                            row.update({k: v for k, v in gaussian.items() if k.startswith("gauss")})
                        row["data_route"] = args.mode
                        seen_uids.add(row["sample_uid"]); content_hashes.add(fingerprint)
                        outputs[frame["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                        report.accepted(row)
                    except (ValueError, KeyError, TypeError) as exc:
                        report.reject(str(exc), input_file=filename, line=line_number)
    result = report.finish(mode=args.mode, index_meta=index.meta,
                           qa_condition="same exact sample_token; no nearest-frame substitutions")
    print(json.dumps(result["counts"]))
    if not any(key.startswith("written:") for key in report.counts):
        raise SystemExit("No matched QA; inspect rejected.jsonl")


if __name__ == "__main__":
    main()
