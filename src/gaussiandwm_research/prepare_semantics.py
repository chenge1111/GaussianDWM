from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel,CLIPProcessor,pipeline


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Produce frozen CLIP text and SAM-region CLIP semantic teachers")
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--data-root",required=True)
    parser.add_argument("--output-manifest",required=True)
    parser.add_argument("--cache-root",required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--text-only",action="store_true")
    parser.add_argument("--feature-height",type=int,default=44)
    parser.add_argument("--feature-width",type=int,default=80)
    parser.add_argument("--max-masks",type=int,default=128)
    args = parser.parse_args()
    root,cache = Path(args.data_root),Path(args.cache_root)
    cache.mkdir(parents=True,exist_ok=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device).eval()
    sam = None if args.text_only else pipeline("mask-generation",model="facebook/sam-vit-base",device=args.device)
    size = (args.feature_height,args.feature_width)
    output = Path(args.output_manifest); output.parent.mkdir(parents=True,exist_ok=True)
    def path(value):
        p = Path(value); return p if p.is_absolute() else root/p
    with output.open("w",encoding="utf-8") as handle:
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            digest = hashlib.sha256(row["query"].encode()).hexdigest()[:24]
            text_path = cache/f"text-{digest}.npy"
            if not text_path.exists():
                inputs = processor(text=[row["query"]],return_tensors="pt",padding=True,truncation=True).to(args.device)
                text = F.normalize(clip.get_text_features(**inputs),dim=-1)[0]
                np.save(text_path,text.cpu().numpy())
            row["clip_text_feature_path"] = str(text_path.resolve())
            if sam is not None:
                feature_paths,id_paths = [],[]
                for vi,image_path in enumerate(row["image_paths"]):
                    image_digest = hashlib.sha256(str(path(image_path).resolve()).encode()).hexdigest()[:24]
                    feature_path = cache/f"sem-{image_digest}.npz"
                    ids_path = cache/f"instance-{image_digest}.npy"
                    if not feature_path.exists() or not ids_path.exists():
                        with Image.open(path(image_path)) as src: image = src.convert("RGB")
                        result = sam(image,points_per_batch=64)
                        masks = [np.asarray(m,dtype=bool) for m in result["masks"]]
                        masks = sorted(masks,key=lambda m:int(m.sum()),reverse=True)
                        if len(masks)>args.max_masks:
                            broad = args.max_masks//2
                            masks = masks[:broad]+masks[-(args.max_masks-broad):]
                        features = torch.zeros(512,*size,device=args.device)
                        ids = torch.full(size,-1,dtype=torch.long,device=args.device)
                        # Large regions first, small objects overwrite their own
                        # mask footprints. No fabricated cross-view identity link.
                        for mi,mask in enumerate(masks):
                            ys,xs = np.where(mask)
                            if len(xs)<16: continue
                            crop = image.crop((int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)))
                            encoded = processor(images=crop,return_tensors="pt").to(args.device)
                            feature = F.normalize(clip.get_image_features(**encoded),dim=-1)[0]
                            down = F.interpolate(torch.from_numpy(mask.copy()).to(args.device).float()[None,None],
                                                 size,mode="nearest")[0,0].bool()
                            features[:,down] = feature[:,None]
                            ids[down] = vi*1_000_000+mi
                        np.savez_compressed(feature_path,features=features.half().cpu().numpy())
                        np.save(ids_path,ids.cpu().numpy())
                    feature_paths.append(str(feature_path.resolve())); id_paths.append(str(ids_path.resolve()))
                row["semantic_feature_paths"],row["instance_id_paths"] = feature_paths,id_paths
                row["instance_id_source"] = "SAM_image_local_not_cross_view_tracks"
            handle.write(json.dumps(row,ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
