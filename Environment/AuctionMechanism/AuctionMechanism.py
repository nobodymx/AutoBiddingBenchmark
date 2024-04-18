import numpy as np


class AuctionMechanism:
    def __init__(self, reserve_pv_price=0.01, min_remaining_budget=0.1, is_sparse=False):
        super().__init__()
        self.reserve_pv_price = reserve_pv_price
        self.min_remaining_budget = min_remaining_budget
        self.is_sparse = is_sparse
        self.slot_coefficients = np.array([1, 0.8, 0.6])  # 竞得坑位系数

    def reset(self):
        pass

    def simulate_ad_bidding(self, pv_values, bids):
        bids[bids < self.reserve_pv_price] = 0
        valid_bids = bids.max(axis=1) > 0  # 去掉全为0的出价
        winner = np.argmax(bids, axis=1)
        winner[bids.max(axis=1) == 0] = -1  # 所有广告主出价为0，本条流量流拍
        valid_winner = np.argmax(bids[valid_bids], axis=1)
        num_pv, num_agent = pv_values[valid_bids].shape
        # 计算reward
        rewards = np.zeros(num_agent)
        if self.is_sparse:
            sparse_rewards = np.random.binomial(n=1, p=np.clip(pv_values, 0, 1), size=pv_values.shape)
            np.add.at(rewards, winner, sparse_rewards[valid_bids][np.arange(num_pv), winner])
        else:
            np.add.at(rewards, valid_winner, pv_values[valid_bids][np.arange(num_pv), valid_winner])
        # 计算市场价格
        market_price = np.partition(bids, -2, axis=1)[:, -2]
        # 避免第二出价为0的情况
        highest_bids = np.max(bids, axis=1)
        market_price = np.where(market_price < highest_bids / 2, highest_bids / 2, market_price)
        # 保留价格 # TODO 保留价如何搞
        market_price[market_price < self.reserve_pv_price] = self.reserve_pv_price
        # 计算花费
        costs = np.zeros(num_agent)
        np.add.at(costs, valid_winner, market_price[valid_bids])
        # TODO 考虑如果所有bids都为0 如何处理
        return rewards, costs, market_price, winner

    def simulate_ad_bidding_multi_pit(self, pv_values, bids):
        num_pv, num_agent = bids.shape
        num_slots = self.slot_coefficients.shape[0]

        # 获取每个PV的前num_slots个最高出价者的索引和出价
        sorted_bid_indices = np.argsort(-bids, axis=1)[:, :num_slots + 1]
        sorted_bids = -np.sort(-bids, axis=1)[:, :num_slots + 1]

        # 获取市场价格（每个流量的市场价格是下一个最高出价*对应坑位的系数，如果没有则为保留价）
        market_prices = sorted_bids[:, 1:num_slots + 1]
        final_cost = market_prices * self.slot_coefficients
        market_prices[market_prices < self.reserve_pv_price] = self.reserve_pv_price
        final_cost[final_cost < self.reserve_pv_price] = self.reserve_pv_price

        # 根据坑位系数计算奖励
        slot_rewards = pv_values[np.arange(num_pv)[:, None], sorted_bid_indices[:, :num_slots]] * self.slot_coefficients

        # 如果是稀疏奖励模式，进行二项分布抽样
        if self.is_sparse:
            slot_rewards = np.random.binomial(n=1, p=np.clip(slot_rewards, 0, 1))

        rewards = np.zeros((num_agent, num_slots))  # 初始化为(num_agent, num_slots)形状的奖励矩阵
        costs = np.zeros((num_agent, num_slots))  # 初始化为(num_agent, num_slots)形状的成本矩阵
        winners = np.full((num_pv, num_slots), -1)  # 用-1初始化

        for slot_index in range(num_slots):
            agent_indices = sorted_bid_indices[:, slot_index]  # 当前坑位的广告主索引
            # 检查当前坑位是否流拍（市场价等于保留价）
            is_unsold = (final_cost[:, slot_index] == self.reserve_pv_price)
            # 对于非流拍的坑位，累加每个广告主的总奖励和成本
            valid_indices = agent_indices[~is_unsold]
            valid_rewards = slot_rewards[~is_unsold, slot_index]
            valid_costs = final_cost[~is_unsold, slot_index]
            np.add.at(rewards, (valid_indices, np.full(valid_indices.shape, slot_index)), valid_rewards)
            np.add.at(costs, (valid_indices, np.full(valid_indices.shape, slot_index)), valid_costs)
            winners[~is_unsold, slot_index] = agent_indices[~is_unsold]
        total_reward = rewards.sum(axis=1)
        total_cost = costs.sum(axis=1)

        return rewards, costs, market_prices, winners
