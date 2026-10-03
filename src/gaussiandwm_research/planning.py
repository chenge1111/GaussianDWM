from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .scene import GaussianScene


class PlanningHead(nn.Module):
    """Predict ego-frame x,y,z,sin(yaw),cos(yaw) and brake/straight/left/right logits."""
    def __init__(self,hidden_size: int,horizon: int = 6):
        super().__init__()
        self.horizon = horizon
        self.trunk = nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,512),nn.SiLU())
        self.trajectory = nn.Linear(512,horizon*5)
        self.command = nn.Linear(512,4)
        self.condition = nn.Sequential(nn.Linear(horizon*5,hidden_size),nn.SiLU(),nn.Linear(hidden_size,hidden_size))
        self.risk_head = nn.Sequential(nn.Conv2d(4,32,3,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d(1),
                                      nn.Flatten(),nn.Linear(32,1))
        self.refine = nn.Sequential(nn.Linear(hidden_size+1,256),nn.SiLU(),nn.Linear(256,horizon*5))
        self.refined_command = nn.Sequential(nn.Linear(hidden_size+horizon*5+1,256),nn.SiLU(),nn.Linear(256,4))

    def forward(self,condition: torch.Tensor):
        x = self.trunk(condition.float())
        raw = self.trajectory(x).reshape(-1,self.horizon,5)
        xy = raw[...,:3].cumsum(1)
        yaw = F.normalize(raw[...,3:5],dim=-1,eps=1e-5)
        trajectory = torch.cat([xy,yaw],-1)
        return trajectory,self.command(x)

    def trajectory_condition(self,trajectory):
        return self.condition(trajectory.flatten(1).float())

    def generated_risk(self,rgb: torch.Tensor,depth: torch.Tensor):
        b,t = rgb.shape[:2]
        x = torch.cat([rgb.float(),depth.float()/80],2).flatten(0,1)
        return self.risk_head(x).reshape(b,t).mean(1,keepdim=True).sigmoid()

    def refine_plan(self,condition,trajectory,risk):
        residual = torch.tanh(self.refine(torch.cat([condition.float(),risk.float()],-1))).reshape_as(trajectory)
        adjusted = trajectory + residual*risk[:,:,None]
        return torch.cat([adjusted[...,:3],F.normalize(adjusted[...,3:5],dim=-1)],-1)

    def refine_commands(self,condition,trajectory,risk):
        return self.refined_command(torch.cat([condition.float(),trajectory.flatten(1).float(),risk.float()],-1))


def collision_risk(trajectory: torch.Tensor,scene: GaussianScene,step_seconds: float = 0.5,
                   radius: float = 2.0,chunk_size: int = 4096) -> torch.Tensor:
    """Differentiable geometric risk proxy, NOT a certified collision detector.

    Restricts obstacle height to [0.25,3m] to exclude most road/sky Gaussians.
    """
    risks = []
    for ti in range(trajectory.shape[1]):
        future = scene.at_time((ti+1)*step_seconds)
        frame = trajectory.new_zeros(trajectory.shape[0])
        for start in range(0,scene.xyz.shape[1],chunk_size):
            xyz = future.xyz[:,start:start+chunk_size]
            valid = future.valid[:,start:start+chunk_size] & (xyz[...,2]>0.25) & (xyz[...,2]<3)
            distance = torch.linalg.vector_norm(xyz[...,:2]-trajectory[:,ti,None,:2],dim=-1)
            weight = future.opacity[:,start:start+chunk_size,0]*valid
            score = torch.sigmoid((radius-distance)*3)*weight
            frame = torch.maximum(frame,score.amax(-1))
        risks.append(frame)
    return torch.stack(risks,1).mean(1)
