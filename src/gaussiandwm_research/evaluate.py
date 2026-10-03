from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Ground-truth planning errors and measured cascade usage")
    parser.add_argument("--predictions",required=True)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    ground = {x["sample_uid"]:x for x in (json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip())}
    rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    ade,fde,latency,token_count,retries = [],[],[],[],[]
    statuses = {}
    for row in rows:
        gt = ground[row["sample_uid"]]
        if row.get("trajectory") is not None and "trajectory" in gt:
            prediction,target = np.asarray(row["trajectory"]),np.asarray(gt["trajectory"])
            if prediction.shape != target.shape:
                raise ValueError("Planning horizon mismatch")
            distances = np.linalg.norm(prediction[:,:2]-target[:,:2],axis=-1)
            ade.append(float(distances.mean())); fde.append(float(distances[-1]))
        latency.append(row["elapsed_seconds"])
        token_count.append(row["total_prefill_tokens"]+row["total_generated_tokens"])
        retries.append(max(0,sum(s["stage"]=="fine" for s in row["sampling_history"])-1))
        statuses[row["status"]] = statuses.get(row["status"],0)+1
    def mean(values): return float(np.mean(values)) if values else None
    output = {"count":len(rows),"planning_ade_xy_m":mean(ade),"planning_fde_xy_m":mean(fde),
              "planning_count":len(ade),"mean_total_tokens":mean(token_count),"mean_resamples":mean(retries),
              "mean_latency_seconds":mean(latency),"statuses":statuses,
              "protocol":"real_gt_planning_and_usage; QA paper metrics require official evaluator"}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(output,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
