from __future__ import annotations
import json
from pathlib import Path
from safetensors.torch import save_file,load_file


def save_delta(model,directory,config,processor,gathered_state=None):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    names = {name for name,p in model.named_parameters() if p.requires_grad}
    # Include running statistics/non-parameter state from the online reconstructor.
    source = model.state_dict() if gathered_state is None else gathered_state
    state = {k:v.detach().cpu().contiguous().clone() for k,v in source.items()
             if k in names or k.startswith("online.")}
    save_file(state,str(directory/"research.safetensors"))
    (directory/"research_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    (directory/"manifest.json").write_text(json.dumps({"format_version":1,"kind":"research_delta",
        "base_model":config["base"],"includes_base_frozen_weights":False},indent=2),encoding="utf-8")
    processor.save_pretrained(directory/"processor")


def load_delta(model,directory):
    state = load_file(str(Path(directory)/"research.safetensors"))
    # Missing frozen base parameters are expected; all delta keys must match.
    unexpected = model.load_state_dict(state,strict=False).unexpected_keys
    if unexpected:
        raise ValueError(f"Unexpected research checkpoint keys: {unexpected[:10]}")
    expected = {name for name,p in model.named_parameters() if p.requires_grad}
    missing = expected-set(state)
    if missing:
        raise ValueError(f"Checkpoint lacks trainable parameters: {sorted(missing)[:10]}")
