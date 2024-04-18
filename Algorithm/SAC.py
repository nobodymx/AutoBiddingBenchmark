import numpy as np
import torch
import torch.nn as nn
from Common.nnModules import MLP, DiagGaussian
from Models.actors import ActorProb
from Models.critics import Critic
from torch.optim.lr_scheduler import CosineAnnealingLR as CALR


class SAC(nn.Module):
    def __init__(
            self,
            params,
            device="cpu",
    ):
        super().__init__()
        self.params = params
        self.actor_backbone = MLP(input_dim=params["obs_dim"],
                                  output_dim=params["actor_bb_out_dim"],
                                  hidden_dims=params["actor_hidden_dim"])
        self.critic2_backbone = MLP(input_dim=params["obs_dim"] + params["action_dim"],
                                    output_dim=params["critic_bb_out_dim"],
                                    hidden_dims=params["critic_hidden_dim"])
        self.critic1_backbone = MLP(input_dim=params["obs_dim"] + params["action_dim"],
                                    output_dim=params["critic_bb_out_dim"],
                                    hidden_dims=params["critic_hidden_dim"])
        self.critic2_target_backbone = MLP(input_dim=params["obs_dim"] + params["action_dim"],
                                           output_dim=params["critic_bb_out_dim"],
                                           hidden_dims=params["critic_hidden_dim"])
        self.critic1_target_backbone = MLP(input_dim=params["obs_dim"] + params["action_dim"],
                                           output_dim=params["critic_bb_out_dim"],
                                           hidden_dims=params["critic_hidden_dim"])
        self.action_dim = params["action_dim"]
        dist = DiagGaussian(
            latent_dim=params["actor_bb_out_dim"],
            output_dim=self.action_dim,
            unbounded=True,
            conditioned_sigma=True
        )
        self._device = device

        self.actor = ActorProb(self.actor_backbone, dist, self._device)
        self.critic1 = Critic(self.critic1_backbone, self._device)
        self.critic2 = Critic(self.critic2_backbone, self._device)
        self.critic1_target = Critic(self.critic1_target_backbone, self._device)
        self.critic2_target = Critic(self.critic2_target_backbone, self._device)

        # hack for jit
        self._tau = 1.0
        self._sync_weight()

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=params["actor_lr"])
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=params["critic_lr"])
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=params["critic_lr"])
        self.actor_sch = CALR(self.actor_optim, params["policy_train_epoch"],
                              eta_min=params["actor_lr"] / params["lr_decay"])
        self.critic1_sch = CALR(self.critic1_optim, params["policy_train_epoch"],
                                eta_min=params["critic_lr"] / params["lr_decay"])
        self.critic2_sch = CALR(self.critic2_optim, params["policy_train_epoch"],
                                eta_min=params["critic_lr"] / params["lr_decay"])

        self._tau = params["tau"]
        self._gamma = params["gamma"]

        if params["auto_alpha"]:
            target_entropy = params["target_entropy"] if params["target_entropy"] \
                else -np.prod(self.action_dim)
            log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=params["alpha_lr"])
            alpha = (target_entropy, log_alpha, alpha_optim)
        else:
            alpha = params["alpha"]

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._eps = np.finfo(np.float32).eps.item()



    def step_scheduler(self):
        for sch in self.schedulers:
            sch.step()

    def train(self, **kwargs):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self):
        for o, n in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def forward(self, obs):
        action, _ = self.actor(obs, deterministic=True)
        squashed_action = torch.tanh(action)
        return squashed_action

    def sample(self, obs):
        action, log_prob = self.actor(obs, deterministic=False)
        squashed_action = torch.tanh(action)
        log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) + self._eps).sum(-1, keepdim=True)
        return squashed_action, log_prob

    def sample_action(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, device=self._device, dtype=torch.float32)
        if deterministic:
            action = self(obs)
        else:
            action, _ = self.sample(obs)
        return action.cpu().detach().numpy()

    def learn(self, data):
        obs, actions, next_obs, terminals, rewards = data["states"], \
                                                     data["actions"], data["next_states"], data["terminals"], \
                                                     data["rewards"]

        rewards = torch.as_tensor(rewards, device=self._device, dtype=torch.float32)
        terminals = torch.as_tensor(terminals, device=self._device, dtype=torch.float32)
        obs = torch.as_tensor(obs, device=self._device, dtype=torch.float32)
        actions = torch.as_tensor(actions, device=self._device, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, device=self._device, dtype=torch.float32)

        # update critic
        q1, q2 = self.critic1(obs, actions).flatten(), self.critic2(obs, actions).flatten()
        with torch.no_grad():
            next_actions, next_log_probs = self.sample(next_obs)
            next_q = torch.min(
                self.critic1_target(next_obs, next_actions), self.critic2_target(next_obs, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards.flatten() + self._gamma * (1 - terminals.flatten()) * next_q.flatten()
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self.sample(obs)
        q1a, q2a = self.critic1(obs, a).flatten(), self.critic2(obs, a).flatten()
        actor_loss = (self._alpha * log_probs.flatten() - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        self._sync_weight()

        return result
