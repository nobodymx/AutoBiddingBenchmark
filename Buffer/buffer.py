from copy import deepcopy
import numpy as np


class Buffer:
    def __init__(self, offline_data):
        self.policy_data = {"offline_data": offline_data,
                            "imaginary_data": {"states": [],
                                               "actions": [],
                                               "rewards": [],
                                               "next_states": [],
                                               "terminals": []}}
        self.imaginary_data_size = 0
        self.offline_data_size = len(self.policy_data["offline_data"])

    def add_imaginary_data(self, states, actions, rewards, next_states, terminals):
        self.policy_data["imaginary_data"]["states"].append(deepcopy(states))
        self.policy_data["imaginary_data"]["actions"].append(deepcopy(actions))
        self.policy_data["imaginary_data"]["rewards"].append(deepcopy(rewards))
        self.policy_data["imaginary_data"]["next_states"].append(deepcopy(next_states))
        self.policy_data["imaginary_data"]["terminals"].append(deepcopy(terminals))
        self.imaginary_data_size += 1

    def sample_offline_data(self, sample_size):
        batch_indices = np.random.randint(0, self.offline_data_size, size=sample_size)
        return {"states": self.policy_data["offline_data"]["states"][batch_indices],
                "actions": self.policy_data["offline_data"]["actions"][batch_indices],
                "rewards": self.policy_data["offline_data"]["rewards"][batch_indices],
                "next_states": self.policy_data["offline_data"]["next_states"][batch_indices],
                "terminals": self.policy_data["offline_data"]["terminals"][batch_indices]}

    def sample_imaginary_data(self, sample_size):
        batch_indices = np.random.randint(0, self.imaginary_data_size, size=sample_size)
        return {"states": np.array(self.policy_data["imaginary_data"]["states"])[batch_indices],
                "actions": np.array(self.policy_data["imaginary_data"]["actions"])[batch_indices],
                "rewards": np.array(self.policy_data["imaginary_data"]["rewards"])[batch_indices],
                "next_states": np.array(self.policy_data["imaginary_data"]["next_states"])[batch_indices],
                "terminals": np.array(self.policy_data["imaginary_data"]["terminals"])[batch_indices]}
