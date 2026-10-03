from __future__ import annotations

from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from gaussiandwm_cvpr.models.qwen_gauss_backbone import BackboneRequest
from .cascade import Selection
from .scene import GaussianScene


class GaussianTokenEncoder(nn.Module):
    """Reuse the upstream aligner; preserve full online semantic gradients."""
    def __init__(self, hidden_size: int, semantic_residual: bool = True):
        super().__init__()
        self.language_compressor = nn.Sequential(nn.Linear(512,64),nn.SiLU(),nn.Linear(64,3))
        self.use_semantic_residual = semantic_residual
        self.semantic_residual = nn.Linear(512,hidden_size,bias=False)
        # Small initial residual leaves the pretrained geometric token scale close
        # to upstream while allowing full CLIP features to carry gradients.
        nn.init.normal_(self.semantic_residual.weight,std=1e-4)

    def pack(self, scene: GaussianScene) -> torch.Tensor:
        code = scene.language_code if scene.language_code is not None else self.language_compressor(scene.language.float())
        return scene.packed(code)

    def forward(self, selection: Selection, aligner: nn.Module) -> torch.Tensor:
        hidden = aligner(selection.values)
        if self.use_semantic_residual:
            hidden = hidden + self.semantic_residual(selection.language.float())
        return hidden.to(next(aligner.parameters()).dtype)


class QwenBridge:
    """Dynamic Gaussian prompt encoding and generation without changing Qwen size."""
    def __init__(self, base: nn.Module, processor: Any, include_images: bool = True,
                 image_max_pixels: int = 224*400):
        self.base,self.processor = base,processor
        self.include_images,self.image_max_pixels = include_images,image_max_pixels

    @property
    def device(self):
        return next(self.base.parameters()).device

    def encode(self, query: str, gaussian_hidden: torch.Tensor,
               images: torch.Tensor | None = None, answer: str | None = None,
               system: str = "", return_labels: bool = False) -> tuple[dict,torch.Tensor | None]:
        if gaussian_hidden.shape[0] != 1:
            raise ValueError("QwenBridge uses one scene per call")
        content,pil_images = [],[]
        if images is not None and self.include_images:
            for image in images[0]:
                array = (image.detach().float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype("uint8")
                pil = Image.fromarray(array)
                ratio = min(1,(self.image_max_pixels/(pil.width*pil.height))**0.5)
                pil = pil.resize((max(28,int(pil.width*ratio)//28*28),max(28,int(pil.height*ratio)//28*28)))
                pil_images.append(pil)
                content.append({"type":"image","image":pil})
        # Use ordinary text rather than a custom 'gauss' chat-template element;
        # this supports both the published template and standard Qwen templates.
        content.append({"type":"text","text":"<|gaussian_start|>" +
                        "<|gaussian_pad|>"*gaussian_hidden.shape[1] + "<|gaussian_end|>\n" + query})
        messages = ([{"role":"system","content":system}] if system else [])
        messages.append({"role":"user","content":content})
        if answer is not None:
            messages.append({"role":"assistant","content":answer})
        text = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=answer is None)
        encoded = self.processor(text=[text],images=pil_images or None,return_tensors="pt",padding=False)
        inputs = {k:v.to(self.device) for k,v in encoded.items() if isinstance(v,torch.Tensor)}
        inputs["selected_gauss_hidden"] = gaussian_hidden
        labels = None
        if return_labels:
            if answer is None:
                raise ValueError("Training labels require a ground-truth answer")
            from gaussiandwm_cvpr.data.dataset import _build_assistant_only_labels
            labels = _build_assistant_only_labels(self.processor.tokenizer,inputs["input_ids"][0])[None]
        return inputs,labels

    def backbone(self, inputs: dict, need_tokens: bool = True):
        return self.base.backbone.forward_backbone(BackboneRequest(
            qwen_inputs=inputs,need_token_hidden=need_tokens,need_global_condition=True,
            task_type="qa",use_gumbel=False))

    @torch.no_grad()
    def generate(self, inputs: dict, max_new_tokens: int = 256) -> tuple[str,dict]:
        """Cached greedy decoding after custom Gaussian prefill.

        Falls back to full-prefix decoding only when the backbone returns no KV
        cache. Total prefill and generated lengths are recorded for efficiency.
        """
        bb = self.backbone(inputs)
        qwen = self.base.backbone._resolve_qwen_module("lm_head")
        hidden = bb.token_hidden_states
        token = qwen.lm_head(hidden[:,-1:].to(qwen.lm_head.weight.dtype)).argmax(-1)
        past = bb.past_key_values
        prefix = inputs["input_ids"]
        mask = inputs["attention_mask"]
        generated = []
        eos = self.processor.tokenizer.eos_token_id
        eos_ids = {eos} if isinstance(eos,int) else set(eos or [])
        im_end = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids.add(im_end)
        for _ in range(max_new_tokens):
            value = int(token[0,-1])
            if value in eos_ids:
                break
            generated.append(value)
            mask = torch.cat([mask,torch.ones_like(token)],1)
            if past is not None:
                # Decode directly with the same multimodal language model that
                # owns the prefill cache. Reuse its multimodal RoPE delta, avoiding
                # wrapper-specific Qwen/Peft rope-delta attribute placement.
                language_model = self.base.backbone._resolve_qwen_module("language_model").language_model
                cache_position = torch.tensor([inputs["input_ids"].shape[1]+len(generated)-1],device=self.device)
                position_ids = cache_position.reshape(1,1,1).expand(3,1,1)
                if bb.rope_deltas is not None:
                    position_ids = position_ids+bb.rope_deltas.reshape(1,1,1).to(position_ids.device)
                causal = language_model(input_ids=token,attention_mask=mask,position_ids=position_ids,
                                        cache_position=cache_position,past_key_values=past,use_cache=True)
                past = causal.past_key_values
                token = qwen.lm_head(causal.last_hidden_state[:,-1:].to(qwen.lm_head.weight.dtype)).argmax(-1)
            else:
                prefix = torch.cat([prefix,token],1)
                expanded = dict(inputs,input_ids=prefix,attention_mask=mask)
                bb = self.backbone(expanded)
                token = qwen.lm_head(bb.token_hidden_states[:,-1:].to(qwen.lm_head.weight.dtype)).argmax(-1)
        result = self.processor.tokenizer.decode(generated,skip_special_tokens=True)
        return result,{"prefill_tokens":inputs["input_ids"].shape[1],"generated_tokens":len(generated),
                       "gaussian_tokens":inputs["selected_gauss_hidden"].shape[1],"cached":past is not None,
                       "_global_condition":self.base.backbone.masked_mean(hidden,inputs["attention_mask"].bool())}
