from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator,DataLoaderConfiguration
from accelerate.utils import set_seed,gather_object,DistributedDataParallelKwargs
from transformers import AutoProcessor,get_cosine_schedule_with_warmup
from .config import load_config
from .data import DrivingManifestDataset,single_scene_collate
from .model import ResearchGaussianDWM
from .checkpoint import save_delta


def main():
    parser = argparse.ArgumentParser(description="Occlusion-aware/online GaussianDWM research training")
    parser.add_argument("--config",default="configs/research/online_joint.yaml")
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--data-root",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--resume",default=None,help="Accelerate state directory for exact optimizer/RNG resume")
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["training"]
    accelerator = Accelerator(mixed_precision=cfg.get("precision","bf16"),
                              gradient_accumulation_steps=cfg.get("gradient_accumulation_steps",2),
                              dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True,data_seed=cfg.get("seed",42)),
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    set_seed(cfg.get("seed",42),device_specific=True)
    root = Path(args.output_dir)
    if accelerator.is_main_process:
        root.mkdir(parents=True,exist_ok=True)
        (root/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
        (root/"environment.json").write_text(json.dumps({"python":platform.python_version(),
            "torch":torch.__version__,"cuda":torch.version.cuda,"world_size":accelerator.num_processes,
            "device":torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "manifest":str(Path(args.manifest).resolve()),"data_root":str(Path(args.data_root).resolve())},indent=2),encoding="utf-8")
    accelerator.wait_for_everyone()
    processor = AutoProcessor.from_pretrained(config["base"]["model_id"],revision=config["base"]["revision"])
    model = ResearchGaussianDWM(config,processor)
    data_cfg = config["reconstruction"]["drivingforward"]["training"]
    dataset = DrivingManifestDataset(args.manifest,args.data_root,data_cfg["height"],data_cfg["width"],config["mode"])
    loader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=cfg.get("num_workers",4),
                        collate_fn=single_scene_collate,pin_memory=True)
    groups = {}
    for name,param in model.named_parameters():
        if not param.requires_grad:
            continue
        rate = cfg.get("lr_lora",1e-5) if "lora_" in name else cfg.get("lr_reconstruction",1e-4) if name.startswith("online.") else cfg.get("lr_new_modules",1e-4)
        groups.setdefault(rate,[]).append(param)
    optimizer = torch.optim.AdamW([{"params":params,"lr":rate} for rate,params in groups.items()],
                                  weight_decay=cfg.get("weight_decay",0.01))
    # Scheduler steps explicitly once per synchronized optimizer update, independent
    # of the number of workers or accumulated microbatches.
    scheduler = get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps=int(cfg["max_steps"]*cfg.get("warmup_ratio",0.03)),num_training_steps=cfg["max_steps"])
    model,optimizer,loader = accelerator.prepare(model,optimizer,loader)
    accelerator.register_for_checkpointing(scheduler)
    step,epoch,skip_batches = 0,0,0
    if args.resume:
        accelerator.load_state(args.resume)
        cursor = json.loads((Path(args.resume)/"cursor.json").read_text(encoding="utf-8"))
        step,epoch,skip_batches = cursor["step"],cursor["epoch"],cursor["batch_index"]+1
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while step < cfg["max_steps"]:
        if hasattr(loader,"set_epoch"):
            loader.set_epoch(epoch)
        current_loader = accelerator.skip_first_batches(loader,skip_batches) if skip_batches else loader
        for batch_index,batch in enumerate(current_loader,start=skip_batches):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    result = model(batch)
                accelerator.backward(result["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(),cfg.get("max_grad_norm",1))
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            step += 1
            if step % cfg.get("logging_steps",20) == 0:
                local_values = {k:float(v.float()) for k,v in result["losses"].items()}
                workers = gather_object([local_values])
                keys = set().union(*(worker.keys() for worker in workers))
                values = {k:sum(worker[k] for worker in workers if k in worker)/sum(k in worker for worker in workers) for k in keys}
                if accelerator.is_main_process:
                    record = {"step":step,"epoch":epoch,"losses":values,**result["telemetry"],
                              "peak_memory_gib":torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0}
                    with (root/"train.jsonl").open("a",encoding="utf-8") as handle:
                        handle.write(json.dumps(record)+"\n")
                    accelerator.print(json.dumps(record))
            if step % cfg.get("save_every_steps",500) == 0:
                path = root/f"checkpoint-{step}"
                accelerator.save_state(path)
                if accelerator.is_main_process:
                    (path/"cursor.json").write_text(json.dumps({"step":step,"epoch":epoch,"batch_index":batch_index}),encoding="utf-8")
            if step >= cfg["max_steps"]:
                break
        epoch += 1
        skip_batches = 0
    # All workers participate in ZeRO gathering; only rank 0 writes the portable delta.
    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        bare = accelerator.unwrap_model(model)
        # Save gathered state (ZeRO2 parameters are replicated, ZeRO3 can be gathered).
        save_delta(bare,root/"final",config,processor,gathered_state=state)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
