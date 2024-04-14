import numpy as np
import torch
from Common.Utils import log_prob
from Common.nnModules import FusionEmbeddingNet, DiagGaussian, ActorProb


class ControlAgents:
    def __init__(self,
                 budgets=None,
                 algo=None,
                 name="ControlAgents",
                 num_agents=10,
                 agent_cate_num=6,
                 num_tick=24,
                 budget_max=2000,
                 pv_num=10000,
                 device="cpu",
                 deterministic=False,
                 policy_mode="shared",
                 obs_dim=10,
                 cate_id_dim=10,
                 embedding_dim=10,
                 embedding_hidden_layers=0,
                 common_hidden_dim=10,
                 id_onehot_output_act_fn="relu",
                 actor_bb_out_dim=8,
                 action_dim=1,
                 agent_cate_list=None,
                 ):
        self.algo = algo
        self.num_agents = num_agents
        self.budget_max = budget_max
        self.budgets = budgets
        self.action_dim = action_dim
        self.agent_cate_num = agent_cate_num
        self.remaining_budgets = self.budgets.astype(np.float64)
        self.num_tick = num_tick
        self.agent_cate_list = agent_cate_list
        self.device = device
        self.pv_num = pv_num
        self.policy_mode = policy_mode
        self.name = name

        self.actor_backbone = FusionEmbeddingNet(dense_input_dim=obs_dim,
                                                 id_onehot_input_dim=cate_id_dim,
                                                 id_onehot_output_dim=embedding_dim,
                                                 id_onehot_hidden_dim=embedding_hidden_layers,
                                                 common_hidden_dim=common_hidden_dim,
                                                 id_onehot_output_act_fn=id_onehot_output_act_fn,
                                                 output_dim=actor_bb_out_dim,
                                                 device=device,
                                                 )

        dist = DiagGaussian(
            latent_dim=actor_bb_out_dim,
            output_dim=self.action_dim,
            unbounded=True,
            conditioned_sigma=True
        )

        self.actor = ActorProb(self.actor_backbone, dist, self.device)
        self.deterministic = deterministic

    def reset(self, policy_mode=None):
        pass

    def sample(self, id_cate_onehot, obs, deterministic=True):
        mu, sigma = self.actor(torch.cat([obs, id_cate_onehot], -1), deterministic)
        if not deterministic:
            action = mu + sigma * torch.normal(torch.zeros(sigma.shape, dtype=sigma.dtype, device=sigma.device),
                                               torch.ones(sigma.shape, dtype=sigma.dtype, device=sigma.device))
        else:
            action = mu
        log_probability = log_prob(action, mu, sigma)
        return action, log_probability

    def take_actions(self, tickIndex, pv_pvalues, history_pv_num):
        bids, obs_lst, id_cate_lst, action_lst, log_prob_lst = [], [], [], [], []
        for agent_index in range(self.num_agents):
            id_cate_onehot, obs = self.norm_state(agent_index, tickIndex, self.remaining_budgets[agent_index],
                                                  history_pv_num)
            obs = torch.Tensor(obs).to(self.device)
            id_cate_onehot = torch.Tensor(id_cate_onehot).to(self.device)
            action, log_probability = self.sample(id_cate_onehot, obs, deterministic=self.deterministic)
            squashed_action = torch.relu(action)
            bids.append(squashed_action.cpu().detach().numpy() * pv_pvalues[:, agent_index])
            obs_lst.append(obs)
            id_cate_lst.append(id_cate_onehot)
            action_lst.append(squashed_action.detach().numpy())
            log_prob_lst.append(log_probability)
        return bids, obs_lst, id_cate_lst, action_lst, log_prob_lst

    def norm_state(self, agent_index, tick_index, remain_budget, pv_num):
        obs = np.array([
            2 * tick_index / self.num_tick - 1,
            2 * remain_budget / self.budgets[agent_index] - 1,
            2 * self.budgets[agent_index] / self.budget_max - 1,
            2 * pv_num / self.pv_num - 1,
        ])
        cate = np.zeros(self.agent_cate_num)
        cate[self.agent_cate_list[agent_index]] = 1

        id_cate_onehot = np.zeros(self.num_agents)
        id_cate_onehot[agent_index] = 1

        id_cate_onehot = np.concatenate([id_cate_onehot, cate])

        return id_cate_onehot, obs

    def update(self, costs):
        self.remaining_budgets -= costs
