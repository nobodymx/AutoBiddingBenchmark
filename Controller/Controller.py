from Agent.RAgent import RAgent
import numpy as np
from Environment.AuctionMechanism.AuctionMechanism import AuctionMechanism
from Environment.PVGenerator.MultiNormalPvGen import MultiNormalPvGen
from Common.Utils import calibration_by_remaining_budget
import math
from collections import OrderedDict
from copy import deepcopy


class Controller:
    def __init__(self,
                 num_controllable_agent=10,
                 num_uncontrollable_agents=20,
                 budget_level=None,
                 num_agent_cate=6,
                 num_tick=24,
                 reserve_pv_price=0.01,
                 min_remaining_budget=0.1,
                 is_sparse=False,
                 pv_num=10000,
                 multi_pit=True,
                 deterministic=False,
                 device="cpu",
                 data_logger=None,
                 control_agents=None,
                 budgets=None,
                 cate=None,
                 ):
        assert budget_level is not None
        self.agent_candidate_list = [RAgent, RAgent, RAgent, RAgent, RAgent]
        self.num_agent_candidate_type = len(self.agent_candidate_list)
        self.num_agent_cate = num_agent_cate
        self.data_logger = data_logger
        self.pv_num = pv_num
        self.device = device
        self.multi_pit = multi_pit
        self.num_controllable_agents = num_controllable_agent
        self.num_uncontrollable_agents = num_uncontrollable_agents
        self.num_agents = self.num_controllable_agents + self.num_uncontrollable_agents
        self.budget_level = budget_level
        self.num_tick = num_tick
        self.deterministic = deterministic

        self.reserve_pv_price = reserve_pv_price
        self.min_remaining_budget = min_remaining_budget
        self.is_sparse = is_sparse
        self.budgets = budgets.astype(np.float64)
        self.remain_budget = self.budgets.copy()
        self.cate = cate
        self.max_budget = max(self.budgets)

        # create agents
        self.uncontrollable_agents = self.create_uncontrollable_agents()
        self.controllable_agents = control_agents
        self.pvGenerator = self.loadPvGenerator()
        self.auctionMechanism = self.loadAuctionMechanism()

    def reset(self):
        self.auctionMechanism.reset()
        self.pvGenerator.reset()
        for i in range(self.num_uncontrollable_agents):
            self.uncontrollable_agents[i].reset()

    def create_uncontrollable_agents(self):
        agents = []
        len_budg_level = len(self.budget_level)
        for index in range(self.num_uncontrollable_agents):
            agents.append(self.agent_candidate_list[np.random.randint(self.num_agent_candidate_type)]
                          (budget=self.budget_level[np.random.randint(len_budg_level)],
                           category=self.cate[index + self.num_controllable_agents]))
        return agents

    def loadPvGenerator(self):
        return MultiNormalPvGen(num_tick=self.num_tick, num_agent_category=self.num_agent_cate,
                                pv_num=self.pv_num,
                                num_caterory=self.num_agent_candidate_type + 1)

    def loadAuctionMechanism(self):
        return AuctionMechanism(reserve_pv_price=self.reserve_pv_price,
                                min_remaining_budget=self.min_remaining_budget,
                                is_sparse=self.is_sparse)

    def create_init_state(self):
        states = np.zeros((self.num_agents, 4 + 6 + self.num_agents))
        for i in range(self.num_agents):
            cate = np.zeros(self.num_agent_cate)
            cate[self.cate[i]] = 1

            id_cate_onehot = np.zeros(self.num_agents)
            id_cate_onehot[i] = 1
            states[i] = np.concatenate([np.array([-1, 1, 2 * self.budgets[i] / self.max_budget - 1, -1]),
                                        cate, id_cate_onehot])
        return states

    def run_simulation(self, num_trajectory=10, save_data=True):
        ratio_max, winner_indices, reward, cost = None, None, 0, 0
        over_cost_ratio = 0
        self.reset()
        data = {}
        for episode in range(num_trajectory):
            rewards, costs = np.zeros(shape=self.num_agents), np.zeros(shape=self.num_agents)  # 记录每个出价智能体获得的reward
            hist_pv_num = 0
            data[episode] = OrderedDict()
            store_states = self.create_init_state()
            for tickIndex in range(self.num_tick):
                store_actions, store_rewards, store_next_states = np.zeros(self.num_agents), np.zeros(
                    self.num_agents), np.zeros((
                    self.num_agents, 4 + 6 + self.num_agents))
                # print(f'episode:{episode}, tick:{tickIndex} begin')
                #  1. 产生流量
                pv_values = self.pvGenerator.pv_values[tickIndex]  # (num_pv,num_agent)
                pv_categorys = self.pvGenerator.pv_categories[tickIndex]  # num_pv
                bids = []
                # 2. 每个智能体出价
                # controllable agents
                controllable_agents_bids, obs_lst, id_cate_lst, action_lst, log_probability_lst = self.controllable_agents.take_actions(
                    tickIndex, pv_values[:, :self.num_controllable_agents], hist_pv_num)
                bids.extend(controllable_agents_bids)
                # record the action
                store_actions[: self.num_controllable_agents] = action_lst
                # uncontrollable agents
                for i, agent in enumerate(self.uncontrollable_agents):
                    if agent.remaining_budget < self.auctionMechanism.min_remaining_budget:
                        bid = np.zeros(shape=(pv_values.shape[0],))
                        alpha = 0
                    else:
                        bid, alpha = agent.action(tickIndex,
                                                  pv_values[:, i + self.num_controllable_agents],
                                                  )
                    bids.append(bid)
                    store_actions[self.num_controllable_agents + i] = alpha

                remaining_budget_list = np.concatenate([self.controllable_agents.remaining_budgets,
                                                        [agent.remaining_budget for agent in
                                                         self.uncontrollable_agents]])
                done_list = (remaining_budget_list < self.auctionMechanism.min_remaining_budget).astype('int')

                bids = np.array(bids).transpose()  # num_pv.num_agent
                # 3. 智能体竞价
                while (ratio_max is None) or ratio_max > 0:
                    if ratio_max and ratio_max > 0:
                        # 防超投机制：一旦发生超投（ratio_max > 0），将超出预算的部分出价置零，并重新参竟。
                        overcost_agent_index_list = np.where(over_cost_ratio > 0)[0]
                        if self.multi_pit:
                            for agent_index in overcost_agent_index_list:
                                for i, coefficient in enumerate(self.auctionMechanism.slot_coefficients):
                                    winner_indices = winner_pit[:, i]
                                    pv_index = np.where(winner_indices == agent_index)[0]
                                    droped_pv_index = np.random.choice(pv_index, math.ceil(
                                        pv_index.shape[0] * over_cost_ratio[agent_index]),
                                                                       replace=False)
                                    bids[droped_pv_index, agent_index] = 0
                        else:
                            for agent_index in overcost_agent_index_list:
                                pv_index = np.where(winner_indices == agent_index)[0]
                                droped_pv_index = np.random.choice(pv_index, math.ceil(
                                    pv_index.shape[0] * over_cost_ratio[agent_index]),
                                                                   replace=False)
                                bids[droped_pv_index, agent_index] = 0
                    if self.multi_pit:
                        reward_pit, cost_pit, market_price_pit, winner_pit = self.auctionMechanism.simulate_ad_bidding_multi_pit(
                            pv_values, bids)
                        reward = reward_pit.sum(axis=1)
                        cost = cost_pit.sum(axis=1)
                        market_price = market_price_pit[:, -1]
                    else:
                        reward, cost, market_price, winner_indices = self.auctionMechanism.simulate_ad_bidding(
                            pv_values, bids)

                    over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 0.0001), 0)
                    ratio_max = over_cost_ratio.max()

                ratio_max = None

                # 4. 根据花费和预算剩余调整reward
                calibrated_reward, calibrated_cost = calibration_by_remaining_budget(reward, remaining_budget_list,
                                                                                     cost, over_cost_ratio)

                # update controllable agents
                self.controllable_agents.update(calibrated_cost[:self.num_controllable_agents])
                for i, agent in enumerate(self.uncontrollable_agents):
                    agent.remaining_budget = agent.remaining_budget - calibrated_cost[i + self.num_controllable_agents]
                rewards += calibrated_reward
                costs += calibrated_cost

                self.remain_budget -= calibrated_cost

                hist_pv_num += len(pv_values)
                # store the reward and next states
                store_rewards = calibrated_reward
                store_next_states = self.make_the_next_state(hist_pv_num, tickIndex + 1)
                if tickIndex == self.num_tick - 1:
                    terminals = np.ones(self.num_agents)
                else:
                    terminals = np.zeros(self.num_agents)

                if save_data:
                    self.data_logger.add_agent_transitions(store_states, store_actions, store_rewards,
                                                           store_next_states, terminals)

                store_states = store_next_states

            self.reset()
        return data, self.data_logger

    def make_the_next_state(self, pv_num, tickIndex):
        states = np.zeros((self.num_agents, 4 + 6 + self.num_agents))
        for i in range(self.num_agents):
            cate = np.zeros(self.num_agent_cate)
            cate[self.cate[i]] = 1
            id_cate_onehot = np.zeros(self.num_agents)
            id_cate_onehot[i] = 1
            states[i] = np.concatenate([np.array([2 * tickIndex / self.num_tick - 1,
                                                  2 * self.remain_budget[i] / self.budgets[i] - 1,
                                                  2 * self.budgets[i] / self.max_budget - 1,
                                                  2 * pv_num / self.pv_num - 1]),
                                        cate, id_cate_onehot])
        return states
