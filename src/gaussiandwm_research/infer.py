from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor
from .data import DrivingManifestDataset,single_scene_collate
from .model import ResearchGaussianDWM
from .checkpoint import load_delta
from .inference import infer_scene


def main():
    parser = argparse.ArgumentParser(description="Bounded cascade/resample and planning-generation inference")
    parser.add_argument("--checkpoint",required=True,help="Portable research delta directory")
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--data-root",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--no-world",action="store_true")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint/"research_config.json").read_text(encoding="utf-8"))
    config["training"]["gradient_checkpointing"] = False
    # The reconstruction initialization checkpoint is unnecessary after loading
    # the portable delta, which includes all online weights and running buffers.
    config["reconstruction"]["weights_dir"] = None
    config["reconstruction"]["drivingforward"]["model"]["weights_init"] = False
    processor = AutoProcessor.from_pretrained(checkpoint/"processor")
    model = ResearchGaussianDWM(config,processor)
    load_delta(model,checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.base.backbone.qwen.config.use_cache = True
    text_config = getattr(model.base.backbone.qwen.config,"text_config",None)
    if text_config is not None:
        text_config.use_cache = True
    dims = config["reconstruction"]["drivingforward"]["training"]
    dataset = DrivingManifestDataset(args.manifest,args.data_root,dims["height"],dims["width"],config["mode"],require_answer=False)
    root = Path(args.output_dir); root.mkdir(parents=True,exist_ok=True)
    with (root/"predictions.jsonl").open("w",encoding="utf-8") as handle:
        for index in range(len(dataset)):
            batch = single_scene_collate([dataset[index]])
            batch = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
                result = infer_scene(model,batch,generate_world=not args.no_world)
            media = result.pop("media")
            if media is not None:
                path = root/f"sample-{index:06d}.npz"
                np.savez_compressed(path,rgb=media["rgb"].cpu().numpy(),depth=media["depth"].cpu().numpy())
                result["media_path"] = path.name
            handle.write(json.dumps(result,ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
