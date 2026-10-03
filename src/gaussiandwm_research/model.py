from __future__ import annotations

from dataclasses import asdict
import json
import torch
from torch import nn
import torch.nn.functional as F
from gaussiandwm_cvpr.models.unified_model import UnifiedGaussianDWM
from gaussiandwm_cvpr.models.world_head import vae_decode_chunked
from gaussiandwm_cvpr.train.trainable_groups import inject_lora_from_config,apply_trainable_groups
from .online import OnlineGaussianReconstructor
from .scene import GaussianScene
from .cascade import CascadedSampler,SceneHint
from .bridge import GaussianTokenEncoder,QwenBridge
from .occlusion import estimate_occlusion,enhance_semantics
from .render import GaussianRenderer
from .planning import PlanningHead,collision_risk
from .losses import masked_l1,PerceptualLoss


COARSE_PROMPT = """Describe the scene using only the available evidence. Return JSON with
summary, complexity (0..1), elements [{name,weight,bounds}], and target_bounds
for the query if grounded. Bounds are [xmin,ymin,zmin,xmax,ymax,zmax] in the
CURRENT ego frame in meters (x forward, y left, z up). Do not invent an unseen
target or its position. Use null target_bounds when unknown. Query: """
FINAL_RULE = """Check whether the sampled Gaussians support the requested target.
If a target is missing but a defensible search region is known, output exactly
[RESAMPLE] {\"bounds\":[xmin,ymin,zmin,xmax,ymax,zmax],\"reason\":\"...\"}.
Coordinates use current ego frame meters, x forward/y left/z up. Otherwise give
the answer and state uncertainty for unsupported details. Do not claim to have
observed a target simply because resampling was requested."""


def planning_cameras(trajectory: torch.Tensor,camera_to_ego: torch.Tensor,view: int = 0):
    b,t,_ = trajectory.shape
    poses = torch.eye(4,device=trajectory.device).expand(b,t,4,4).clone()
    sine,cosine = trajectory[...,3],trajectory[...,4]
    poses[...,0,0],poses[...,0,1] = cosine,-sine
    poses[...,1,0],poses[...,1,1] = sine,cosine
    poses[...,:3,3] = trajectory[...,:3]
    return poses @ camera_to_ego[:,view:view+1].float()


class ResearchGaussianDWM(nn.Module):
    def __init__(self,config: dict,processor):
        super().__init__()
        self.config = config
        base_cfg = config["base"]
        self.base = UnifiedGaussianDWM.from_pretrained(base_cfg["model_id"],revision=base_cfg["revision"])
        self.base = inject_lora_from_config(self.base,config["lora"])
        groups = ["gauss_aligner_core","qwen_backbone_lora","qa_lm_head_lora"]
        if config.get("world",{}).get("enabled",False):
            groups += ["cond_fusion","world_unet_lora"]
        apply_trainable_groups(self.base,groups)
        h = self.base.backbone.hidden_size
        self.tokens = GaussianTokenEncoder(h,config.get("semantic_residual",True))
        self.sampler = CascadedSampler(h,**config["sampling"])
        self.online = OnlineGaussianReconstructor(config["reconstruction"]) if config["mode"] == "online" else None
        self.renderer = GaussianRenderer()
        self.planner = PlanningHead(h,config["planning"]["horizon"])
        self.bridge = QwenBridge(self.base,processor,**config.get("prompt",{}))
        self.perceptual = PerceptualLoss() if config["loss"].get("perceptual",0)>0 else None
        self.base.world_head.bundle.vae.requires_grad_(False)
        self.base.cond_fusion.image_encoder.requires_grad_(False)
        training = config.get("training",{})
        if training.get("gradient_checkpointing",True):
            qwen = self.base.backbone.qwen
            if hasattr(qwen,"gradient_checkpointing_enable"):
                qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            if hasattr(qwen,"enable_input_require_grads"):
                qwen.enable_input_require_grads()
            unet = self.base.world_head.bundle.unet
            if hasattr(unet,"enable_gradient_checkpointing"):
                unet.enable_gradient_checkpointing()
            qwen.config.use_cache = False

    def train(self,mode=True):
        super().train(mode)
        self.base.world_head.bundle.vae.eval()
        self.base.cond_fusion.image_encoder.eval()
        return self

    def make_scene(self,batch: dict) -> GaussianScene:
        if self.online is not None:
            return self.online(batch)
        if "offline_scene" in batch:
            return batch["offline_scene"]
        packed = batch["gauss_values"]
        language = self.base.backbone.gauss_aligner.decode_language_clip_features(packed)
        shape = packed.shape[:2]
        return GaussianScene(packed[...,:3],packed[...,3:6],packed[...,6:10],packed[...,10:11],
                             torch.zeros_like(packed[...,:3]),language,
                             torch.ones(shape,dtype=torch.bool,device=packed.device),
                             batch.get("gauss_instance_ids",torch.full(shape,-1,dtype=torch.long,device=packed.device)),
                             batch.get("gauss_view_ids",torch.zeros(shape,dtype=torch.long,device=packed.device)),
                             language_code=packed[...,11:14])

    def prepare_scene(self,batch):
        scene = self.make_scene(batch)
        flags = self.config.get("occlusion",{})
        if flags.get("enabled",True):
            _,_,_,height,width = batch["images"].shape
            occluded = estimate_occlusion(scene,batch["intrinsics"],batch["camera_to_ego"],height,width,
                                         threshold=flags.get("threshold",0.5),depth_margin=flags.get("depth_margin",0.5))
        else:
            occluded = torch.zeros_like(scene.valid)
        if flags.get("semantic_fill",True):
            scene = enhance_semantics(scene,occluded,neighbors=flags.get("neighbors",8))
            # Recompress enhanced semantics rather than retain stale offline code.
            if scene.language_code is not None:
                from dataclasses import replace
                compressed = self.tokens.language_compressor(scene.language.float())
                code = torch.where(occluded[...,None],compressed,scene.language_code)
                scene = replace(scene,language_code=code)
        return scene,occluded

    def forward(self,batch: dict) -> dict:
        if batch["images"].shape[0] != 1:
            raise ValueError("Variable-budget research training uses per-device batch=1; accumulate for larger batches")
        scene,occluded = self.prepare_scene(batch)
        reconstruction_scene = scene
        if self.config.get("detach_downstream_scene",False):
            scene = scene.detached()
        packed = self.tokens.pack(scene)
        hint = SceneHint.parse(json.dumps(batch.get("scene_hint",{})))
        coarse = self.sampler.coarse(scene,packed)
        coarse_tokens = self.tokens(coarse,self.base.backbone.gauss_aligner)
        # Teacher hints supervise coarse output only when annotations provide them.
        # No fabricated scene counts/bounds are used as ground truth.
        coarse_answer = json.dumps(batch["scene_hint"]) if batch.get("scene_hint") else None
        losses = {}
        if self.sampler.cascade_enabled:
            inputs,labels = self.bridge.encode(COARSE_PROMPT+batch["query"],coarse_tokens,batch["images"],
                                               answer=coarse_answer,return_labels=coarse_answer is not None)
            coarse_bb = self.bridge.backbone(inputs)
            coarse_condition = self._prompt_condition(coarse_bb,inputs,labels)
            if labels is not None and self.config["loss"].get("coarse",0)>0:
                losses["coarse"] = self.base.qa_head(token_hidden_states=coarse_bb.token_hidden_states,
                                                  lm_head=self.base.backbone.qwen.lm_head,labels=labels).loss
        else:
            coarse_condition = None
            hint = SceneHint()
        fine = self.sampler.fine(scene,packed,batch["clip_text_feature"],hint,coarse_condition,
                                 occluded,batch.get("task_kind","global"))
        hidden = self.tokens(fine,self.base.backbone.gauss_aligner)
        query = batch["query"]+"\nGlobal context: "+hint.summary
        inputs,labels = self.bridge.encode(query,hidden,batch["images"],answer=batch["answer"],
                                           system=FINAL_RULE,return_labels=True)
        bb = self.bridge.backbone(inputs)
        losses["qa"] = self.base.qa_head(token_hidden_states=bb.token_hidden_states,
                                        lm_head=self.base.backbone.qwen.lm_head,labels=labels).loss
        condition = self._prompt_condition(bb,inputs,labels)
        trajectory,commands = self.planner(condition)
        if self.config["planning"].get("enabled",True) and "trajectory" in batch:
            losses["planning"] = F.smooth_l1_loss(trajectory,batch["trajectory"].float())
            if "command" in batch:
                losses["command"] = F.cross_entropy(commands,batch["command"].long())
            losses["collision"] = collision_risk(trajectory,scene).mean()
        # Compression consistency trains the 512 -> 3 code against the published
        # frozen/pretrained semantic decoder rather than leaving it arbitrary.
        if self.config["loss"].get("semantic_code",0)>0:
            decoded = self.base.backbone.gauss_aligner.ae_decoder(packed[...,11:14].to(
                next(self.base.backbone.gauss_aligner.parameters()).dtype))
            losses["semantic_code"] = (1-F.cosine_similarity(decoded.float(),scene.language.float(),dim=-1)).mean()
        if self.online is not None:
            self._reconstruction_losses(batch,reconstruction_scene,losses)
        if self.config["world"].get("enabled",False):
            self._world_losses(batch,scene,condition,trajectory,losses)
        total = sum(self.config["loss"].get(name,0)*value for name,value in losses.items())
        return {"loss":total,"losses":{k:v.detach() for k,v in losses.items()},
                "telemetry":{"coarse_tokens":coarse.used_tokens if self.sampler.cascade_enabled else 0,"fine_tokens":fine.used_tokens,
                             "occluded_gaussians":int(occluded.sum()),"region_empty":fine.region_empty}}

    @staticmethod
    def _prompt_condition(bb,inputs,labels):
        mask = inputs["attention_mask"].bool()
        if labels is not None:
            positions = torch.arange(labels.shape[1],device=labels.device)[None]
            first_answer = positions.expand_as(labels).masked_fill(labels == -100,labels.shape[1]).amin(1)
            mask = mask & (positions < first_answer[:,None])
        return (bb.token_hidden_states*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)

    def _reconstruction_losses(self,batch,scene,losses):
        h,w = batch["images"].shape[-2:]
        rendered = self.renderer(scene,batch["intrinsics"],batch["camera_to_ego"],h,w)
        losses["reconstruction_rgb"] = F.l1_loss(rendered["rgb"],batch["images"].float())
        if "depth" in batch:
            mask = (batch["depth"]>0)&torch.isfinite(batch["depth"])
            losses["reconstruction_depth"] = masked_l1(rendered["depth"],batch["depth"],mask)
        if "semantic_features" in batch:
            target = batch["semantic_features"].float()
            sh,sw = target.shape[-2:]
            k = batch["intrinsics"].clone().float()
            k[...,0,:] *= sw/w; k[...,1,:] *= sh/h
            semantic = self.renderer(scene,k,batch["camera_to_ego"],sh,sw,render_language=True)
            # Teacher masks supervise even where predicted alpha is low, so the
            # encoder cannot avoid semantic supervision by reducing opacity.
            mask = batch.get("semantic_mask",target.abs().sum(2,keepdim=True)>0)
            similarity = 1-F.cosine_similarity(semantic["language"],target,dim=2)
            valid = mask.squeeze(2).bool()
            losses["semantic"] = similarity[valid].mean() if valid.any() else similarity.sum()*0
        if "future_images" in batch and "future_camera_to_ego" in batch:
            # Future camera transforms are expressed in CURRENT ego coordinates.
            steps = batch["future_images"].shape[1]
            future = []
            for ti in range(steps):
                future.append(self.renderer(scene,batch["intrinsics"],batch["future_camera_to_ego"][:,ti],
                                             h,w,seconds=(ti+1)*self.config["planning"]["step_seconds"])["rgb"])
            losses["motion_rgb"] = F.l1_loss(torch.stack(future,1),batch["future_images"].float())

    def render_conditions(self,batch,scene,trajectory):
        cfg = self.config["world"]
        height,width,view = cfg["height"],cfg["width"],cfg["view_index"]
        k = batch["intrinsics"][:,view:view+1].clone().float()
        ih,iw = batch["images"].shape[-2:]
        k[...,0,:] *= width/iw; k[...,1,:] *= height/ih
        cameras = planning_cameras(trajectory,batch["camera_to_ego"],view)
        rgb,depth = [],[]
        for ti in range(trajectory.shape[1]):
            render = self.renderer(scene,k,cameras[:,ti:ti+1],height,width,
                                   seconds=(ti+1)*self.config["planning"]["step_seconds"])
            rgb.append(render["rgb"][:,0]); depth.append(render["depth"][:,0])
        return torch.stack(rgb,1),torch.stack(depth,1)

    def world_target(self,batch):
        cfg = self.config["world"]
        # Train generation on actual future RGB+depth, never the pseudo conditions.
        rgb = batch["future_images"][:,:,cfg["view_index"]]
        depth = batch["future_depth"][:,:,cfg["view_index"]]
        size = (cfg["height"],cfg["width"])
        rgb = F.interpolate(rgb.flatten(0,1),size,mode="bilinear",align_corners=False).reshape(1,-1,3,*size)
        depth = F.interpolate(depth.flatten(0,1),size,mode="nearest").reshape(1,-1,1,*size)
        normalized_depth = (depth/cfg["depth_max"]).clamp(0,1)
        vae = self.base.world_head.bundle.vae
        dtype = next(vae.parameters()).dtype
        with torch.no_grad():
            def encode(x):
                z = vae.encode((x.flatten(0,1)*2-1).to(dtype)).latent_dist.mode()
                return (z*self.base.world_head.vae_scaling_factor).reshape(1,x.shape[1],*z.shape[1:])
            latent = {"rgb_latents":encode(rgb),"depth_latents":encode(normalized_depth.expand(-1,-1,3,-1,-1))}
        return latent,rgb,depth

    def _world_losses(self,batch,scene,condition,trajectory,losses):
        if "future_images" not in batch or "future_depth" not in batch:
            raise ValueError("Joint world training requires future RGB and metric depth ground truth")
        cfg = self.config["world"]
        rgb,depth = self.render_conditions(batch,scene,trajectory)
        latent,gt_rgb,gt_depth = self.world_target(batch)
        world_condition = condition + self.planner.trajectory_condition(trajectory).to(condition.dtype)
        ref = F.interpolate(batch["images"][:,cfg["view_index"]],(cfg["height"],cfg["width"]),
                            mode="bilinear",align_corners=False)*2-1
        cond = self.base.cond_fusion(ref_pixel_values=ref,text_cond_seed=world_condition)
        output = self.base.world_head(target_latents=latent,pseudo_pixel_values=rgb*2-1,
                                      pseudo_depth_values=(depth/cfg["depth_max"]).clamp(0,1).expand(-1,-1,3,-1,-1)*2-1,
                                      cond_embeddings=cond,world_meta={"fps":cfg["fps"],"motion_bucket_id":127})
        losses["diffusion"] = output.loss
        z = output.denoised_latents
        vae = self.base.world_head.bundle.vae
        # Frozen VAE weights, but NOT a no_grad decode: perceptual and output MSE
        # flow through the denoiser and layout renderer back to reconstruction.
        def decode(latents):
            x = vae_decode_chunked(vae,latents.flatten(0,1).to(next(vae.parameters()).dtype),
                                   self.base.world_head.vae_scaling_factor,decode_chunk_size=cfg.get("decode_chunk_size",2))
            return ((x.float()+1)/2).reshape(1,latents.shape[1],*x.shape[1:])
        predicted = decode(z[:,:,:4])
        predicted_depth = decode(z[:,:,4:]).mean(2,keepdim=True)*cfg["depth_max"]
        losses["generation_rgb"] = F.mse_loss(predicted,gt_rgb)
        losses["generation_depth"] = masked_l1(predicted_depth,gt_depth,gt_depth>0)
        if self.perceptual is not None:
            losses["perceptual"] = self.perceptual(predicted,gt_rgb)
        if self.config["planning"].get("enabled",True):
            risk = self.planner.generated_risk(predicted,predicted_depth)
            target_risk = collision_risk(trajectory.detach(),scene).detach()[:,None]
            losses["risk"] = F.binary_cross_entropy(risk,target_risk)
            refined = self.planner.refine_plan(condition,trajectory,risk)
            if "trajectory" in batch:
                losses["refined_planning"] = F.smooth_l1_loss(refined,batch["trajectory"].float())
            if "command" in batch:
                refined_commands = self.planner.refine_commands(condition,refined,risk)
                losses["refined_command"] = F.cross_entropy(refined_commands,batch["command"].long())
