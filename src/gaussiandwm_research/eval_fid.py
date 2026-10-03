from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance


def main():
    parser = argparse.ArgumentParser(description="FID over exported RGB clips; match real/fake frame protocol externally")
    parser.add_argument("--real-dir",required=True,help="NPZ files containing float RGB [B,T,3,H,W] in [0,1]")
    parser.add_argument("--fake-dir",required=True)
    parser.add_argument("--device",default="cuda")
    args = parser.parse_args()
    metric = FrechetInceptionDistance(feature=2048,normalize=True).to(args.device)
    counts = []
    for directory,real in [(args.real_dir,True),(args.fake_dir,False)]:
        count = 0
        for path in sorted(Path(directory).glob("*.npz")):
            with np.load(path,allow_pickle=False) as data:
                rgb = torch.from_numpy(data["rgb"]).float()
            frames = rgb.reshape(-1,*rgb.shape[-3:])
            for part in frames.split(32):
                metric.update(part.to(args.device).clamp(0,1),real=real)
            count += len(frames)
        counts.append(count)
    if min(counts)<2:
        raise ValueError("FID needs at least two real and fake frames")
    print({"FID":float(metric.compute()),"real_frames":counts[0],"fake_frames":counts[1],
           "protocol":"torchmetrics/torch-fidelity Inception2048; report resolution/frame selection"})


if __name__ == "__main__":
    main()
