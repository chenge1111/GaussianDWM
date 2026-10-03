from __future__ import annotations
from pathlib import Path
import copy
import yaml


def merge(base,patch):
    result = copy.deepcopy(base)
    for key,value in patch.items():
        result[key] = merge(result.get(key,{}),value) if isinstance(value,dict) else copy.deepcopy(value)
    return result


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = config.pop("extends",None)
    return merge(load_config(path.parent/parent),config) if parent else config
