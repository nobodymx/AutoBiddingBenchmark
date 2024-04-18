import numpy as np
from copy import deepcopy


class DataLogger:
    def __init__(self, num_agents, train_ratio=0.7, represent_index=0):
        self.num_agent = num_agents
        self.data = [self._meta_data() for _ in range(self.num_agent)]
        self.train_ratio = train_ratio
        self.data_num = 0
        self.represent_index = represent_index

    def _meta_data(self):
        return {"states": None,
                "actions": None,
                "rewards": None,
                "next_states": None,
                "terminals": None}

    def save(self):
        for i in range(self.num_agent):
            np.save("Data/" + str(i) + "_states.npy", self.data[i]["states"])
            np.save("Data/" + str(i) + "_actions.npy", self.data[i]["actions"])
            np.save("Data/" + str(i) + "_rewards.npy", self.data[i]["rewards"])
            np.save("Data/" + str(i) + "_next_states.npy", self.data[i]["next_states"])
            np.save("Data/" + str(i) + "_terminals.npy", self.data[i]["terminals"])

    def load(self, path="Data/"):
        for i in range(self.num_agent):
            self.data[i]["states"] = np.load(path + str(i) + "_states.npy")
            self.data[i]["actions"] = np.load(path + str(i) + "_actions.npy")
            self.data[i]["rewards"] = np.load(path + str(i) + "_rewards.npy")
            self.data[i]["next_states"] = np.load(path + str(i) + "_next_states.npy")
            self.data[i]["terminals"] = np.load(path + str(i) + "_terminals.npy")
        self.data_num = len(self.data[0]["states"])

    def generate_train_data(self):
        train_data = {"x": [],
                      "y": [],
                      "ground_truth": []}

        for i in range(self.num_agent):
            train_data["x"].append(np.concatenate([self.data[i]["states"],
                                                   self.data[i]["actions"]], 1))
            train_data["ground_truth"].append(np.concatenate([self.data[i]["next_states"],
                                                              self.data[i]["rewards"]], 1))
            y = []
            for j in range(self.num_agent):
                if j != i:
                    y.append(deepcopy(np.concatenate([self.data[j]["states"],
                                                      self.data[j]["actions"]], 1)))

            train_data["y"].append(np.stack(y, 1))
        train_data["x"] = np.concatenate(train_data["x"], 0)
        train_data["y"] = np.concatenate(train_data["y"], 0)
        train_data["ground_truth"] = np.concatenate(train_data["ground_truth"], 0)
        return train_data

    def add_agent_transitions(self, states, actions, rewards, next_states, terminals):
        for agent_index in range(self.num_agent):
            prev_data = self.data[agent_index]
            if prev_data["states"] is None:
                prev_data["states"] = np.expand_dims(states[agent_index], 0)
                prev_data["actions"] = np.expand_dims([actions[agent_index]], 0)
                prev_data["rewards"] = np.expand_dims([rewards[agent_index]], 0)
                prev_data["next_states"] = np.expand_dims(next_states[agent_index], 0)
                prev_data["terminals"] = np.expand_dims(terminals[agent_index], 0)
            else:
                prev_data["states"] = np.concatenate([prev_data["states"], np.expand_dims(states[agent_index], 0)])
                prev_data["actions"] = np.concatenate([prev_data["actions"], np.expand_dims([actions[agent_index]], 0)])
                prev_data["rewards"] = np.concatenate([prev_data["rewards"], np.expand_dims([rewards[agent_index]], 0)])
                prev_data["next_states"] = np.concatenate(
                    [prev_data["next_states"], np.expand_dims(next_states[agent_index], 0)])
                prev_data["terminals"] = np.concatenate(
                    [prev_data["terminals"], np.expand_dims(terminals[agent_index], 0)])
            self.data[agent_index] = prev_data

    def add_agent_transition(self, agent_index, state, action, reward, next_state, terminals):
        prev_data = self.data[agent_index]
        if prev_data["states"] is None:
            prev_data["states"] = np.array(state)
            prev_data["actions"] = np.array(action)
            prev_data["rewards"] = np.array(reward)
            prev_data["next_states"] = np.array(next_state)
            prev_data["terminals"] = np.array(terminals)
        else:
            prev_data["states"] = np.concatenate([prev_data["states"], state])
            prev_data["actions"] = np.concatenate([prev_data["actions"], action])
            prev_data["rewards"] = np.concatenate([prev_data["rewards"], reward])
            prev_data["next_states"] = np.concatenate([prev_data["next_states"], next_state])
            prev_data["terminals"] = np.concatenate([prev_data["terminals"], terminals])
        self.data[agent_index] = prev_data
