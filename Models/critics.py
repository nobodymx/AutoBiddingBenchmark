import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, backbone, device="cpu"):
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone
        latent_dim = getattr(backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1)
        self.to()

    def to(self):
        self.backbone.to(self.device)
        self.last.to(self.device)

    def forward(self, obs, actions):
        # obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        actions = actions.flatten(1)
        # actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
        obs = torch.cat([obs, actions], dim=1)
        logits = self.backbone(obs)
        values = self.last(logits)
        return values


class CriticID(nn.Module):
    def __init__(self, backbone, device="cpu", dense_dim=11, bb_out_dim=50):
        super().__init__()
        self.device = torch.device(device)
        self.backbone = backbone
        self.dense_dim = dense_dim
        self.bb_out_dim = bb_out_dim
        self.last = nn.Linear(self.bb_out_dim, 1)
        self.to()

    def to(self):
        self.backbone.to(self.device)
        self.last.to(self.device)

    def forward(self, obs, actions):
        # obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        actions = actions.flatten(1)
        # actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
        dense_x = torch.cat([obs[:, :self.dense_dim], actions], dim=1)
        id_x = obs[:, self.dense_dim:]
        obs = torch.cat([dense_x, id_x], dim=1)
        logits = self.backbone(obs)
        values = self.last(logits)
        return values
