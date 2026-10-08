from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import re
from .public_data import (FrameIndex, Report, annotation_sources, decode_records,
                         source_split, CAMERAS)


def plain(value):
    if hasattr(value, "tolist"): return plain(value.tolist())
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): raise ValueError("Nonfinite annotation value")
    return value


def group_name(family, name, record):
    category = str(record.get("category", "")).lower()
    name = name.lower()
    if family == "omnidrive":
        parts = name.replace("\\", "/").split("/")
        if "desc" in parts or "description" in record: return "Omni_Desc", "global"
        if "conv" in parts: return "Omni_Conv", "global"
        return "Omni_VQA", "global"
    if "gemini_caption" in record: return "NuInteract_Caption", "global"
    if "3d visual grounding" in name: return "NuInteract_3DVG", "local"
    if "planning" in name or category == "planning": return "NuInteract_Planning", "global"
    if "2d visual grounding" in name or category in {"2d_perception_location", "2d_visual_grounding"}:
        return "NuInteract_2DVG", "local"
    if "region description" in name: return "NuInteract_RD&P", "local"
    return "NuInteract_Other", "global"


def text_value(value):
    if isinstance(value, str): return value
    if isinstance(value, (list, dict, int, float)): return json.dumps(plain(value), ensure_ascii=False)
    raise ValueError("Missing or unsupported QA text value")


def qa_pairs(record, family, policy):
    captions = record.get("gemini_caption")
    if isinstance(captions, dict):
        for view in CAMERAS:
            if captions.get(view):
                yield f"caption:{view}", f"Describe the road scene in <{view}>.", text_value(captions[view]), {}, []
        if captions.get("FRONT") and captions.get("BACK"):
            yield "caption:overall", "Describe the road scene surrounding the ego vehicle.", (
                text_value(captions["FRONT"]) + "\n" + text_value(captions["BACK"])), {}, []
        return
    if "description" in record:
        yield "description", "Describe the road scene surrounding the ego vehicle.", text_value(record["description"]), {}, []
        return
    if "query" in record or "question" in record:
        question = text_value(record.get("query", record.get("question")))
        answer = text_value(record.get("answer"))
        yield "0", question, answer, record, []
        return
    conversations = record.get("conversations", record.get("messages"))
    if not isinstance(conversations, list): raise ValueError("Unsupported annotation conversation schema")
    history, pending, emitted = [], None, False
    for index, message in enumerate(conversations):
        if not isinstance(message, dict): raise ValueError("Conversation turn must be a mapping")
        role = message.get("role", message.get("from"))
        value = text_value(message.get("content", message.get("value")))
        if role in {"user", "human"}:
            if pending is not None: raise ValueError("Consecutive unanswered user turns")
            pending = (index, value, message)
        elif role in {"assistant", "gpt"}:
            if pending is None: raise ValueError("Assistant turn has no matching question")
            turn, question, question_meta = pending
            metadata = dict(message)
            if any(marker in question for marker in ("<embeding>", "<embedding>")):
                question = insert_numeric_embedding(question, question_meta)
            context = [] if policy == "independent" else list(history)
            yield str(turn), question, value, metadata, context
            rendered_answer = insert_numeric_embedding(value, metadata) if any(
                marker in value for marker in ("<embeding>", "<embedding>")) else value
            history.extend([{"role": "user", "content": question},
                            {"role": "assistant", "content": rendered_answer}])
            pending, emitted = None, True
        elif role == "system":
            history.append({"role": "system", "content": value})
        else:
            raise ValueError(f"Unsupported conversation role {role}")
    if pending is not None: raise ValueError("Unanswered final user turn")
    if not emitted: raise ValueError("No paired QA turns")


def insert_numeric_embedding(text, metadata):
    structured = {}
    for field in ("bboxs_3d_seq", "bboxs_2d_seq", "ids"):
        if field in metadata: structured[field] = plain(metadata[field])
    if not any(field in structured for field in ("bboxs_3d_seq", "bboxs_2d_seq")):
        raise ValueError("Embedding placeholder has no real numeric box annotation")
    rendered = json.dumps(structured, ensure_ascii=False, allow_nan=False)
    # Upstream may serialize one array holding all boxes as one embedding block.
    # Multiple ambiguous blocks are not assigned guessed per-object correspondences.
    count = text.count("<embeding>") + text.count("<embedding>")
    if count != 1: raise ValueError("Multiple embedding placeholders need explicit correspondence")
    return text.replace("<embeding>", rendered).replace("<embedding>", rendered)


def grounding_answer(answer, metadata, group, query, structured):
    if "<embeding>" in answer or "<embedding>" in answer:
        return insert_numeric_embedding(answer, metadata), "numeric_embedding_inserted_source_coordinates"
    if structured and group == "NuInteract_2DVG":
        # Parse only the released <box>(x1,y1),(x2,y2)</box> syntax.
        pattern = r"<CAM_[A-Z_]+>|<box>\s*\(([-+\d.eE]+),\s*([-+\d.eE]+)\),\s*\(([-+\d.eE]+),\s*([-+\d.eE]+)\)\s*</box>"
        camera = None
        query_views = set(re.findall(r"CAM_[A-Z_]+", query))
        if len(query_views) == 1: camera = next(iter(query_views))
        boxes = []
        for match in re.finditer(pattern, answer):
            if match.group(0).startswith("<CAM_"):
                camera = match.group(0)[1:-1]
            else:
                if camera not in CAMERAS: raise ValueError("2D box has no unambiguous camera label")
                values = [float(match.group(i)) for i in range(1, 5)]
                if not all(math.isfinite(v) for v in values): raise ValueError("Nonfinite 2D box")
                boxes.append({"camera": camera, "bbox_2d": values})
        if boxes:
            return json.dumps({"boxes": boxes}, ensure_ascii=False), "2D_xyxy_original_pixel_coordinates"
        if "<box>" in answer: raise ValueError("Unparsed 2D box syntax")
    return answer, "source_text_preserved"


def main():
    parser = argparse.ArgumentParser(description="Convert released NuInteract/OmniDrive QA into exact SDK-matched JSONL")
    parser.add_argument("--source", choices=["nuinteract", "omnidrive"], required=True)
    parser.add_argument("--annotations", nargs="+", required=True, help="Directories or JSON/PKL/ZIP/TAR sources")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-map", help="Optional explicit identity aliases/record overrides, never scene-order guesses")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--conversation-policy", choices=["source", "history", "independent"], default="source")
    parser.add_argument("--grounding-format", choices=["structured", "preserve"], default="structured")
    parser.add_argument("--source-split-policy", choices=["require", "sdk-only"], default="require")
    parser.add_argument("--fail-on-reject", action="store_true")
    args = parser.parse_args()
    index = FrameIndex(args.index)
    aliases = json.loads(Path(args.sample_map).read_text(encoding="utf-8")) if args.sample_map else {}
    report = Report(args.output_dir)
    policy = ("independent" if args.source == "omnidrive" else "history") if args.conversation_policy == "source" else args.conversation_policy
    seen = set()
    with ExitStack() as stack:
        outputs = {split: stack.enter_context((Path(args.output_dir) / f"{split}.qa.jsonl").open("w", encoding="utf-8")) for split in ("train", "val")}
        for name, stream in annotation_sources(args.annotations):
            try:
                origin_split = source_split(name)
                for row_id, record in decode_records(stream, name):
                    report.counts["input_records"] += 1
                    try:
                        if not isinstance(record, dict): raise ValueError("Record must be a mapping")
                        frame, evidence = index.resolve(record, name, str(row_id), aliases)
                        split = frame["split"]
                        if split not in outputs or args.split not in {"all", split}:
                            report.counts["outside_requested_split"] += 1; continue
                        if origin_split and origin_split != split and args.source_split_policy == "require":
                            raise ValueError("Public source split disagrees with official SDK scene split")
                        if set(frame["cameras"]) != set(CAMERAS): raise ValueError("SDK frame lacks six camera keyframes")
                        group, kind = group_name(args.source, name, record)
                        for turn, question, raw_answer, metadata, history in qa_pairs(record, args.source, policy):
                            if not question.strip() or not raw_answer.strip(): raise ValueError("Empty QA pair")
                            answer, answer_format = grounding_answer(raw_answer, metadata, group, question,
                                                                     args.grounding_format == "structured")
                            # Keep camera tags and pixel coordinates, remove only
                            # generic image placeholders replaced by the new prompt.
                            question = question.replace("<image>", "").strip()
                            query = question
                            if history:
                                query = "Conversation context:\n" + "\n".join(f"{m['role']}: {m['content']}" for m in history) + "\nCurrent question: " + question
                            fingerprint = hashlib.sha256(json.dumps([frame["sample_token"], group, query, answer],
                                ensure_ascii=False).encode()).hexdigest()
                            if fingerprint in seen:
                                report.counts["exact_duplicates"] += 1; continue
                            seen.add(fingerprint)
                            row = {"sample_uid": f"{args.source}-{fingerprint[:24]}",
                                   "sample_token": frame["sample_token"], "scene_token": frame["scene_token"],
                                   "scene_name": frame["scene_name"], "frame_index": frame["frame_index"],
                                   "split": split, "query": query, "answer": answer, "task_kind": kind,
                                   "qa_group": group, "qa_subtask": str(record.get("category", "")),
                                   "question": question, "raw_answer": raw_answer,
                                   "image_paths": [frame["cameras"][v]["filename"] for v in CAMERAS],
                                   "source": {"dataset": args.source, "file": name, "record_id": str(row_id),
                                   "turn": turn, "source_split": origin_split, "identity_evidence": evidence,
                                   "conversation_policy": policy, "answer_format": answer_format}}
                            annotation = {field: plain(metadata[field]) for field in ("bboxs_3d_seq", "bboxs_2d_seq", "ids") if field in metadata}
                            if "bbox" in record: annotation["bbox"] = plain(record["bbox"])
                            if annotation: row["grounding_annotation"] = annotation
                            outputs[split].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                            report.accepted(row)
                    except (ValueError, TypeError, KeyError, AttributeError) as exc:
                        report.reject(str(exc), source_file=name, record_id=str(row_id))
            except Exception as exc:
                report.reject("source_file_unreadable", source_file=name, error=str(exc))
    result = report.finish(source=args.source, sdk_meta=index.meta, conversation_policy=policy,
                           source_split_policy=args.source_split_policy,
                           protocol="public_source_reconstructed_QA; not author_processed_GaussianDWM_annotations")
    print(json.dumps({"counts": result["counts"], "sample_counts": result["sample_counts"]}))
    if not sum(value for key, value in report.counts.items() if key.startswith("written:")):
        raise SystemExit("No matched QA written; inspect rejected.jsonl")
    if args.fail_on_reject and any(key.startswith("rejected:") for key in report.counts):
        raise SystemExit("Rejected records exist; outputs retained for review")


if __name__ == "__main__":
    main()
