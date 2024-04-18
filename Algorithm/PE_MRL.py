from Models.transition_model import TransitionModel
from Algorithm.SAC import SAC
from Common.functional import shuffle, dict_batch_generator
from tqdm import tqdm
import numpy as np
import torch
from Buffer.buffer import Buffer
from Agent.RAgent import RAgent
from copy import deepcopy


class PEMRL:
    def __init__(self, env_params, policy_params, data, env_train_data, budgets,
                 env_train_epoch=100, policy_train_epoch=100, rollout_freq=10,
                 rollout_size=1, num_agent=30, rollout_length=1,
                 representative_index=0, obs_dim=4,
                 real_ratio=0.7,
                 policy_train_batch=300,
                 background_agent_num=29,
                 ):
        self.offline_data = data
        self.offline_data_length = len(self.offline_data[0]["states"])
        self.env_train_data = env_train_data
        self.env_train_epoch = env_train_epoch
        self.policy_train_epoch = policy_train_epoch
        self.rollout_freq = rollout_freq
        self.rollout_size = rollout_size
        self.budgets = budgets
        self.num_agent = num_agent
        self.rollout_length = rollout_length
        self.represent_index = representative_index
        self.obs_dim = obs_dim
        self.real_ratio = real_ratio
        self.policy_train_batch = policy_train_batch
        self.background_agent_num = background_agent_num
        self.background_agents = [RAgent(budget=self.budgets[i]) for i in range(self.background_agent_num)]

        self.env_params = env_params
        self.policy_params = policy_params

        self.env_model = TransitionModel(input_dim=env_params["input_dim"],
                                         output_dim=env_params["output_dim"],
                                         reward_penalty_coeff=env_params["reward_penalty_coeff"],
                                         decay_weights=env_params["decay_weights"],
                                         mlp_hidden_layer=env_params["mlp_hidden_layer"],
                                         trans_mlp_hidden_layers=env_params["trans_mlp_hidden_layers"],
                                         total_mlp_hidden_layers=env_params["total_mlp_hidden_layers"],
                                         if_id_cate=False,
                                         )
        self.policy = SAC(policy_params)
        self.buffer = Buffer(self.offline_data[self.represent_index])

    def train(self, learn_dynamic=False, learn_policy_dynamic=True):
        if learn_dynamic:
            self.learn_dynamics()
        if learn_policy_dynamic:
            self.learn_policy()

    def learn_dynamics(self):
        train_data = shuffle(self.env_train_data)
        # train model
        model_train_iters = 0

        for epoch in range(self.env_train_epoch):
            for train_data_batch in dict_batch_generator(train_data, self.env_params["env_model_batch_size"]):
                model_log_infos = self.env_model.update(train_data_batch)  # main training function
                model_train_iters += 1

    def learn_policy(self):
        # train loop
        num_time_steps = 0
        for e in range(self.policy_train_epoch):
            self.policy.train()
            with tqdm(total=self.policy_train_epoch, desc=f"Epoch #{e}/{self.policy_train_epoch}") as t:
                while t.n < t.total:
                    if num_time_steps % self.rollout_freq == 0:
                        self.rollout()
                    # update policy by sac
                    loss = self.update_policy()
                    t.set_postfix(**loss)
                    num_time_steps += 1
                    t.update(1)

    def update_policy(self):
        real_sample_size = int(self.offline_data_length * self.real_ratio)
        fake_sample_size = min(self.policy_train_batch - real_sample_size,
                               self.buffer.imaginary_data_size)
        real_batch = self.buffer.sample_offline_data(real_sample_size)
        fake_batch = self.buffer.sample_imaginary_data(fake_sample_size)

        data = {"states": np.concatenate(
            [real_batch["states"][:, :self.obs_dim].reshape(-1, self.obs_dim),
             fake_batch["states"][:, :self.obs_dim].reshape(-1, self.obs_dim)], 0),
            "actions": np.concatenate([real_batch["actions"].reshape(-1, 1), fake_batch["actions"].reshape(-1, 1)],
                                      0),
            "rewards": np.concatenate([real_batch["rewards"].reshape(-1, 1), fake_batch["rewards"].reshape(-1, 1)],
                                      0),
            "next_states": np.concatenate([real_batch["next_states"][:, :self.obs_dim].reshape(-1, self.obs_dim),
                                           fake_batch["next_states"][:, :self.obs_dim].reshape(-1, self.obs_dim)], 0),
            "terminals": np.concatenate(
                [real_batch["terminals"].reshape(-1, 1), fake_batch["terminals"].reshape(-1, 1)], 0)}
        loss = self.policy.learn(data)
        return loss

    def rollout(self):
        init_transitions = self._sample_initial_transitions()
        # rollout
        states = [init_transitions[i]["states"][0] for i in range(self.num_agent)]

        for rollout_step in range(self.rollout_length):
            actions = []
            index = 0
            for i in range(self.num_agent):
                if i == self.represent_index:
                    action, log_prob = self.policy.sample(torch.Tensor(states[self.represent_index][:self.obs_dim]))
                    actions.append(action.detach().numpy())
                else:
                    bids, alpha = self.background_agents[index].action(rollout_step)
                    actions.append(deepcopy(np.array([alpha])))
                    index += 1
            actions = np.concatenate(actions)

            rep_state_action = np.expand_dims(
                np.concatenate([states[self.represent_index], np.array([actions[self.represent_index]])]), 0)
            back_state_action = np.concatenate(
                [np.array(states[self.represent_index + 1:]), np.expand_dims(actions[self.represent_index + 1:], 1)], 1)
            back_state_action = np.expand_dims(back_state_action, 0)
            pred_next_policy_input, pred_rewards, terminals, penalty = self.env_model.penalty_predict(rep_state_action,
                                                                                                      back_state_action)
            self.buffer.add_imaginary_data(states[self.represent_index], actions[self.represent_index], pred_rewards,
                                           pred_next_policy_input, terminals)

            if terminals == 1:
                break
            x, y = [], []
            for i in range(self.num_agent):
                x.append(np.concatenate([states[i], np.array([actions[i]])], 0))
                temp_y = []
                for j in range(self.num_agent):
                    if j != i:
                        temp_y.append(deepcopy(np.concatenate([states[i], np.array([actions[i]])], 0)))
                y.append(deepcopy(np.array(temp_y)))
            y = np.stack(y, 0)
            x = np.array(x)
            states = self.env_model.predict(x, y)

    def _sample_initial_transitions(self):
        batch_indices = np.random.randint(0, self.offline_data_length, size=self.rollout_size)
        return [{"states": self.offline_data[i]["states"][batch_indices],
                 "actions": self.offline_data[i]["actions"][batch_indices],
                 "rewards": self.offline_data[i]["rewards"][batch_indices],
                 "next_states": self.offline_data[i]["next_states"][batch_indices]}
                for i in range(self.num_agent)]
