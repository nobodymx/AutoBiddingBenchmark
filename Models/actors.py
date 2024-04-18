import torch
import torch.nn as nn


# for PPO/A2C/SAC
class ActorProb(nn.Module):
    def __init__(self, backbone, dist_net, device="cpu"):
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone
        self.dist_net = dist_net
        self.to()

    def get_dist(self, obs):
        # obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist

    def to(self):
        self.backbone.to(self.device)
        self.dist_net.to(self.device)

    def forward(self, obs, deterministic: bool):
        logits = self.backbone(obs)
        return self.dist_net(logits, deterministic)
