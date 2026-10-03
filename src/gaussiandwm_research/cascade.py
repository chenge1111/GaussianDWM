from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import torch
from torch import nn
import torch.nn.functional as F
from .scene import GaussianScene


@dataclass
class SceneHint:
    summary: str = ""
    complexity: float = 0.5
    # Elements: {name, weight, bounds: [xmin,ymin,zmin,xmax,ymax,zmax]}.
    elements: list[dict] = field(default_factory=list)
    target_bounds: list[float] | None = None

    @classmethod
    def parse(cls, text: str) -> SceneHint:
        try:
            body = text[text.index("{"):text.rindex("}")+1]
            data = json.loads(body)
            def bounds(value):
                if not isinstance(value, list) or len(value) != 6:
                    return None
                x = [float(a) for a in value]
                return x if all(math.isfinite(a) for a in x) and all(x[i]<x[i+3] for i in range(3)) else None
            elements = []
            for item in data.get("elements", [])[:64]:
                region = bounds(item.get("bounds"))
                weight = float(item.get("weight", 1))
                if region is not None and math.isfinite(weight):
                    elements.append({"name":str(item.get("name","object")),
                                     "weight":max(weight,0), "bounds":region})
            complexity = float(data.get("complexity", 0.5))
            if not math.isfinite(complexity):
                complexity = 0.5
            return cls(str(data.get("summary", "")), min(max(complexity,0),1),
                       elements, bounds(data.get("target_bounds")))
        except (ValueError, TypeError, KeyError, AttributeError):
            return cls(summary=text[:2048])


@dataclass
class Selection:
    values: torch.Tensor              # [1,K,14]
    language: torch.Tensor            # [1,K,512]
    indices: torch.Tensor             # [K], anchor ids
    requested_budget: int
    used_tokens: int
    region_empty: bool = False


def region_mask(xyz: torch.Tensor, bounds: list[float], expansion: float = 0) -> torch.Tensor:
    box = xyz.new_tensor(bounds)
    return ((xyz >= box[:3]-expansion) & (xyz <= box[3:]+expansion)).all(-1)


def uniform_indices(scene: GaussianScene, budget: int) -> torch.Tensor:
    """Deterministic view-stratified coverage, without opacity filtering."""
    valid = torch.where(scene.valid[0])[0]
    if valid.numel() == 0:
        raise ValueError("Scene contains no valid Gaussians")
    budget = min(budget, valid.numel())
    groups = [valid[scene.view_ids[0,valid] == v] for v in torch.unique(scene.view_ids[0,valid])]
    selected = []
    for group in groups:
        k = min(len(group), max(1, budget//len(groups)))
        selected.append(group[torch.linspace(0,len(group)-1,k,device=valid.device).long()])
    ids = torch.unique(torch.cat(selected), sorted=True)
    if len(ids) > budget:
        ids = ids[torch.linspace(0,len(ids)-1,budget,device=valid.device).long()]
    elif len(ids) < budget:
        available = valid[~torch.isin(valid,ids)]
        extra = available[torch.linspace(0,len(available)-1,budget-len(ids),device=valid.device).long()]
        ids = torch.cat([ids,extra])
    return ids


class CascadedSampler(nn.Module):
    """Task-conditioned cross-attention scoring plus adaptive element quotas.

    Upstream similarity scoring is retained as a residual. Training relaxes hard
    selection over spatial neighbors. Anchor choice, budgets and prompts remain
    discrete; continuous scene/token paths are differentiable, not the JSON loop.
    """
    def __init__(self, hidden_size: int, coarse_tokens: int = 1024,
                 min_tokens: int = 2048, max_tokens: int = 4096,
                 occlusion_boost: float = 2.0, temperature: float = 0.2,
                 neighbors: int = 32, chunk_size: int = 64,
                 adaptive: bool = True, soft_training: bool = True,
                 boost_enabled: bool = True, cascade_enabled: bool = True,
                 attention_enabled: bool = True):
        super().__init__()
        self.coarse_tokens, self.min_tokens, self.max_tokens = coarse_tokens,min_tokens,max_tokens
        self.occlusion_boost, self.temperature = occlusion_boost,temperature
        self.neighbors, self.chunk_size = neighbors,chunk_size
        self.adaptive, self.soft_training, self.boost_enabled = adaptive,soft_training,boost_enabled
        self.cascade_enabled,self.attention_enabled = cascade_enabled,attention_enabled
        self.query = nn.Linear(512,128)
        self.key = nn.Linear(512,128)
        self.global_query = nn.Linear(hidden_size,128)
        nn.init.zeros_(self.global_query.weight)
        nn.init.zeros_(self.global_query.bias)

    def coarse(self, scene: GaussianScene, packed: torch.Tensor) -> Selection:
        idx = uniform_indices(scene, self.coarse_tokens)
        return Selection(packed[:,idx], scene.language[:,idx], idx, self.coarse_tokens, len(idx))

    def fine(self, scene: GaussianScene, packed: torch.Tensor, text_feature: torch.Tensor,
             hint: SceneHint, global_condition: torch.Tensor | None,
             occluded: torch.Tensor, task_kind: str = "global", retry: int = 0) -> Selection:
        if scene.xyz.shape[0] != 1:
            raise ValueError("Sampler processes one scene at a time for variable budgets")
        if task_kind not in {"global", "local"}:
            raise ValueError("task_kind must be global or local")
        lang = F.normalize(scene.language[0].float(),dim=-1)
        text = F.normalize(text_feature.reshape(-1).float(),dim=-1)
        q = self.query(text)
        if global_condition is not None:
            q = q + self.global_query(global_condition.reshape(-1).float())
        score = lang @ text
        if self.attention_enabled:
            score = score + (self.key(lang)*q).sum(-1)/math.sqrt(q.numel())
        if self.boost_enabled:
            # Multiply positive attention weights by 2, rather than doubling
            # negative cosine scores (which would suppress occluded points).
            score = score + occluded[0].float()*math.log(self.occlusion_boost)
        valid = scene.valid[0].clone()
        empty = False
        if task_kind == "local" and hint.target_bounds is not None:
            valid &= region_mask(scene.xyz[0], hint.target_bounds, expansion=retry*2.0)
            empty = not bool(valid.any())
            if empty:
                # Global recovery is explicit in telemetry; do not pretend the
                # requested target region was covered when it was empty.
                valid = scene.valid[0].clone()
        requested = (int(round((self.min_tokens + hint.complexity*(self.max_tokens-self.min_tokens))/256))*256
                     if self.adaptive else self.max_tokens)
        requested = min(self.max_tokens, max(self.min_tokens,requested+retry*256))
        k = min(requested,int(valid.sum()))
        if k <= 0:
            raise ValueError("No candidates available for fine sampling")
        score = score.masked_fill(~valid, -torch.inf)
        # Assign quotas to semantic elements on global tasks. Remaining slots use
        # global attention. Unavailable/overlapping slots are backfilled uniquely.
        chosen = torch.empty(0,dtype=torch.long,device=score.device)
        if task_kind == "global" and hint.elements:
            total = sum(e["weight"] for e in hint.elements) or 1
            for element in hint.elements:
                mask = valid & region_mask(scene.xyz[0],element["bounds"])
                mask[chosen] = False
                quota = min(max(1,int(0.8*k*element["weight"]/total)),int(mask.sum()), k-len(chosen))
                if quota > 0:
                    chosen = torch.cat([chosen,score.masked_fill(~mask,-torch.inf).topk(quota).indices])
        if len(chosen) < k:
            remaining = score.clone()
            remaining[chosen] = -torch.inf
            chosen = torch.cat([chosen,remaining.topk(k-len(chosen)).indices])
        if self.training and self.soft_training:
            vals, langs = [], []
            candidates = torch.where(valid)[0]
            for anchors in chosen.split(self.chunk_size):
                # Sparse local soft attention limits memory to chunk_size x N.
                distances = torch.cdist(scene.xyz[0,anchors].float(),scene.xyz[0,candidates].float())
                dist, near = distances.topk(min(self.neighbors,len(candidates)),largest=False)
                support = candidates[near]
                weights = F.softmax((score[support]-dist)/self.temperature,dim=-1)
                vals.append((packed[0,support]*weights[...,None]).sum(1))
                langs.append((scene.language[0,support]*weights[...,None]).sum(1))
            values, language = torch.cat(vals)[None], torch.cat(langs)[None]
            values = torch.cat([values[...,:6],F.normalize(values[...,6:10],dim=-1),values[...,10:]],-1)
        else:
            values, language = packed[:,chosen], scene.language[:,chosen]
        return Selection(values,F.normalize(language,dim=-1),chosen,requested,k,empty)
