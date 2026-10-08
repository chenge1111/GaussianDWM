"""CPU-side public annotation IO and evidence-based nuScenes identity matching."""
from __future__ import annotations

from collections import Counter, defaultdict, OrderedDict
import hashlib
import importlib
import io
import json
from pathlib import Path
import pickle
import re
import tarfile
import zipfile

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK"]
TOKEN = re.compile(r"^[0-9a-f]{32}$")


def path_key(value: str) -> str:
    value = str(value).replace("\\", "/")
    for marker in ("samples/", "sweeps/"):
        position = value.find(marker)
        if position >= 0:
            return value[position:]
    return value.lstrip("./")


def json_rows(path: Path):
    """Stream large SDK tables when ijson is installed; normal JSON otherwise."""
    try:
        import ijson
    except ImportError:
        yield from json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("rb") as stream:
            yield from ijson.items(stream, "item", use_float=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


class AnnotationUnpickler(pickle.Unpickler):
    """Read plain annotation containers and NumPy arrays without arbitrary globals."""
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) in {("numpy", "dtype"), ("numpy", "ndarray"),
                              ("numpy.core.multiarray", "_reconstruct"),
                              ("numpy.core.multiarray", "scalar"),
                              ("numpy._core.multiarray", "_reconstruct"),
                              ("numpy._core.multiarray", "scalar")}:
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"Unsupported annotation pickle global: {module}.{name}")


def records_from_object(value):
    if isinstance(value, list):
        yield from enumerate(value)
    elif isinstance(value, dict):
        if any(key in value for key in ("conversations", "query", "question", "gemini_caption", "description")):
            yield "0", value
        elif isinstance(value.get("data"), list):
            yield from enumerate(value["data"])
        else:
            for key, record in value.items():
                if isinstance(record, dict):
                    if TOKEN.fullmatch(str(key)) and not any(k in record for k in ("token", "sample_token")):
                        record = dict(record, token=str(key))
                    yield key, record
                elif isinstance(record, list):
                    for index, row in enumerate(record):
                        if isinstance(row, dict) and TOKEN.fullmatch(str(key)):
                            row = dict(row, sample_token=str(key))
                        yield f"{key}:{index}", row
    else:
        raise ValueError(f"Unsupported annotation root {type(value).__name__}")


def decode_records(stream, name):
    suffix = Path(name).suffix.lower()
    if suffix == ".jsonl":
        for index, line in enumerate(stream):
            if line.strip():
                yield index, json.loads(line)
    elif suffix in {".pkl", ".pickle"}:
        yield from records_from_object(AnnotationUnpickler(stream).load())
    else:
        yield from records_from_object(json.load(stream))


def annotation_sources(paths):
    """Yield (source identifier, byte stream); archives are never extracted."""
    for argument in paths:
        root = Path(argument)
        files = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in files:
            if not path.is_file() or path.name == "token_name.json":
                continue
            if path.suffix.lower() in {".json", ".jsonl", ".pkl", ".pickle"}:
                with path.open("rb") as stream:
                    yield str(path.resolve()), stream
            elif path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    for member in sorted(archive.namelist()):
                        if Path(member).suffix.lower() in {".json", ".jsonl", ".pkl", ".pickle"} and Path(member).name != "token_name.json":
                            with archive.open(member) as stream:
                                yield f"{path.resolve()}::{member}", stream
            elif path.name.endswith((".tar.gz", ".tgz", ".tar")):
                with tarfile.open(path, "r|*") as archive:
                    for member in archive:
                        if member.isfile() and Path(member.name).suffix == ".json" and Path(member.name).name != "token_name.json":
                            with archive.extractfile(member) as stream:
                                yield f"{path.resolve()}::{member.name}", stream


def source_split(name: str) -> str | None:
    parts = name.split("::")[-1].replace("\\", "/").lower().split("/")
    # Innermost dataset directory wins over an unrelated ancestor named train.
    for part in reversed(parts[:-1]):
        if part in {"train", "training"}: return "train"
        if part in {"val", "validation", "test", "testing"}: return "val"
    return None


def image_references(record):
    result = []
    def collect(value):
        if isinstance(value, str) and value.lower().endswith((".jpg", ".jpeg", ".png")):
            result.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value: collect(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key not in {"OVERALL", "overall"}: collect(item)
    for key in ("cam_path", "image", "images", "image_paths", "image_path", "filename", "cams"):
        if key in record: collect(record[key])
    return result


class FrameIndex:
    def __init__(self, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported nuScenes frame index format")
        self.meta = payload["meta"]
        self.samples = {row["sample_token"]: row for row in payload["frames"]}
        self.sd_to_sample, self.image_to_sample = {}, {}
        for token, row in self.samples.items():
            for view, camera in row["cameras"].items():
                self.sd_to_sample[camera["sample_data_token"]] = token
                key = path_key(camera["filename"])
                if key in self.image_to_sample and self.image_to_sample[key] != token:
                    raise ValueError(f"Ambiguous SDK image path {key}")
                self.image_to_sample[key] = token

    def resolve(self, record, source_name="", record_id="", aliases=None):
        aliases = aliases or {}
        evidence = []
        for key in ("sample_token", "token", "sample_idx", "sample_id", "sample_data_token"):
            raw = record.get(key)
            if isinstance(raw, str):
                mapped = aliases.get(raw, raw)
                token = mapped if mapped in self.samples else self.sd_to_sample.get(mapped)
                if token: evidence.append((f"field:{key}", token))
        # OmniDrive filenames are sample_token.json. Never treat arbitrary QA ids
        # or directory scene numbers as a sample token.
        stem = Path(source_name.split("::")[-1]).stem
        if stem in self.samples:
            evidence.append(("source_filename", stem))
        for image in image_references(record):
            token = self.image_to_sample.get(path_key(image))
            if token: evidence.append(("image_path", token))
            elif "samples/" in path_key(image) or "sweeps/" in path_key(image):
                raise ValueError(f"Image reference is not a camera keyframe in this SDK version: {image}")
        override = aliases.get(f"{source_name}#{record_id}")
        if override:
            if override not in self.samples: raise ValueError("Sample override is not in SDK index")
            evidence.append(("explicit_sample_map", override))
        candidates = {token for _, token in evidence}
        if len(candidates) != 1:
            raise ValueError("Conflicting sample identity evidence" if candidates else "No exact sample token or SDK camera image match")
        token = next(iter(candidates))
        row = self.samples[token]
        scene_token = record.get("scene_token")
        if scene_token and scene_token != row["scene_token"]:
            raise ValueError("Annotation scene_token disagrees with SDK sample")
        return row, [key for key, _ in evidence]


class Report:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.counts = Counter()
        self.groups = Counter()
        self.scenes = defaultdict(set)
        self.samples = defaultdict(set)
        self.rejects = (self.root / "rejected.jsonl").open("w", encoding="utf-8")

    def reject(self, reason, **context):
        self.counts[f"rejected:{reason}"] += 1
        self.rejects.write(json.dumps({"reason": reason, **context}, ensure_ascii=False) + "\n")

    def accepted(self, row):
        split = row["split"]
        self.counts[f"written:{split}"] += 1
        self.groups[f"{split}:{row.get('qa_group', 'unknown')}"] += 1
        self.scenes[split].add(row["scene_token"])
        self.samples[split].add(row["sample_token"])

    def finish(self, **metadata):
        self.rejects.close()
        payload = {**metadata, "counts": dict(self.counts), "groups": dict(self.groups),
                   "scenes": {k: sorted(v) for k, v in self.scenes.items()},
                   "sample_counts": {k: len(v) for k, v in self.samples.items()}}
        payload["train_val_scene_overlap"] = sorted(self.scenes["train"] & self.scenes["val"])
        (self.root / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
