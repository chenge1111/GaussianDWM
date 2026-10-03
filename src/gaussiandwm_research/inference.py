from __future__ import annotations

import json
import time
import torch
import torch.nn.functional as F
from .cascade import SceneHint
from .model import COARSE_PROMPT,FINAL_RULE
from .planning import collision_risk


def parse_resample(text: str) -> list[float] | None:
    if not text.strip().startswith("[RESAMPLE]"):
        return None
    try:
        data = json.loads(text.strip()[len("[RESAMPLE]"):].strip())
        return SceneHint.parse(json.dumps({"target_bounds":data["bounds"]})).target_bounds
    except (ValueError,TypeError,KeyError):
        return None


@torch.no_grad()
def infer_scene(model,batch: dict,generate_world: bool = True) -> dict:
    started = time.perf_counter()
    scene,occluded = model.prepare_scene(batch)
    packed = model.tokens.pack(scene)
    coarse = model.sampler.coarse(scene,packed)
    hidden = model.tokens(coarse,model.base.backbone.gauss_aligner)
    if model.sampler.cascade_enabled:
        inputs,_ = model.bridge.encode(COARSE_PROMPT+batch["query"],hidden,batch["images"])
        summary,usage = model.bridge.generate(inputs,model.config["inference"]["coarse_max_new_tokens"])
        hint = SceneHint.parse(summary)
        global_condition = usage.pop("_global_condition")
        telemetry = [{"stage":"coarse",**usage}]
    else:
        summary,hint,global_condition,telemetry = "",SceneHint(),None,[]
    answer,status,condition = "","complete",global_condition
    max_retries = model.config["inference"]["max_resamples"]
    for retry in range(max_retries+1):
        selection = model.sampler.fine(scene,packed,batch["clip_text_feature"],hint,global_condition,
                                       occluded,"local" if retry else batch.get("task_kind","global"),retry)
        hidden = model.tokens(selection,model.base.backbone.gauss_aligner)
        inputs,_ = model.bridge.encode(batch["query"]+"\nGlobal context: "+hint.summary,hidden,batch["images"],system=FINAL_RULE)
        answer,usage = model.bridge.generate(inputs,model.config["inference"]["max_new_tokens"])
        condition = usage.pop("_global_condition")
        telemetry.append({"stage":"fine","retry":retry,"region_empty":selection.region_empty,
                          "requested_budget":selection.requested_budget,**usage})
        bounds = parse_resample(answer)
        if not answer.strip().startswith("[RESAMPLE]"):
            break
        if bounds is None:
            status = "invalid_resample_request"
            break
        if retry == max_retries:
            status = "resample_exhausted"
            break
        hint.target_bounds = bounds
    trajectory,commands = model.planner(condition)
    planning_enabled = model.config["planning"].get("enabled",True)
    plan_history = []
    generated = None
    if generate_world and model.config["world"].get("enabled",False):
        cfg = model.config["world"]
        for cycle in range(model.config["inference"].get("max_plan_refinements",1)+1):
            rgb,depth = model.render_conditions(batch,scene,trajectory)
            world_condition = condition+model.planner.trajectory_condition(trajectory).to(condition.dtype)
            ref = F.interpolate(batch["images"][:,cfg["view_index"]],(cfg["height"],cfg["width"]),
                                mode="bilinear",align_corners=False)*2-1
            cond = model.base.cond_fusion(ref_pixel_values=ref,text_cond_seed=world_condition)
            generated = model.base.world_head.generate(
                pseudo_pixel_values=rgb*2-1,
                pseudo_depth_values=(depth/cfg["depth_max"]).clamp(0,1).expand(-1,-1,3,-1,-1)*2-1,
                cond_embeddings=cond,world_meta={"fps":cfg["fps"],"motion_bucket_id":127},
                num_inference_steps=model.config["inference"].get("world_steps",25),
                guidance_scale=model.config["inference"].get("guidance_scale",2.0))
            video = ((generated["rgb"].float()+1)/2).clamp(0,1)
            metric_depth = ((generated["depth"].float()+1)/2)*cfg["depth_max"]
            risk = model.planner.generated_risk(video,metric_depth)
            geometric = collision_risk(trajectory,scene)[:,None]
            combined = torch.maximum(risk,geometric)
            plan_history.append({"cycle":cycle,"risk":float(combined[0,0]),
                                 "trajectory":trajectory[0].cpu().tolist()})
            generated = {"rgb":video,"depth":metric_depth}
            if float(combined[0,0]) < model.config["inference"].get("risk_threshold",0.5):
                break
            if cycle < model.config["inference"].get("max_plan_refinements",1):
                trajectory = model.planner.refine_plan(condition,trajectory,combined)
                commands = model.planner.refine_commands(condition,trajectory,combined)
    # Command is the original trained command head; conservative brake correction
    # is explicit when the bounded generation loop still predicts high risk.
    command = int(commands.argmax(-1)[0])
    if plan_history and plan_history[-1]["risk"] >= model.config["inference"].get("risk_threshold",0.5):
        command = 0
    return {"sample_uid":batch["sample_uid"],"answer":answer,"status":status,
            "coarse_summary":summary,"trajectory":trajectory[0].cpu().tolist() if planning_enabled else None,
            "command":["brake","straight","left","right"][command] if planning_enabled else None,
            "planning_history":plan_history,"sampling_history":telemetry,
            "total_prefill_tokens":sum(x["prefill_tokens"] for x in telemetry),
            "total_generated_tokens":sum(x["generated_tokens"] for x in telemetry),
            "elapsed_seconds":time.perf_counter()-started,"media":generated,
            "risk_protocol":"learned_generated_risk_plus_geometric_proxy"}
