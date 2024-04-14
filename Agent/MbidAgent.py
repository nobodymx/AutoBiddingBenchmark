import numpy as np
from Agent.BaseAgent import BaseAgent


class MbidAgent(BaseAgent):
    def __init__(self, budget=100, name="MbidAgent", category=0, roi=1,
                 num_tick=24, bid_scale=0.24):
        super().__init__()
        self.budget = budget
        self.remaining_budget = budget
        self.num_tick = num_tick
        self.bid_scale = bid_scale
        self.base_actions = np.ones(num_tick) * bid_scale
        self.name = name
        self.category = category
        self.roi = roi

    def reset(self):
        self.remaining_budget = self.budget

    def action(self, tickIndex, pv_pvalues, ):
        shape_of_pv_values = pv_pvalues.shape
        bids = np.full(shape_of_pv_values, self.base_actions[tickIndex])
        bids = bids + np.random.normal(0, 0.01, size=shape_of_pv_values)
        bids[bids < 0] = 0
        return bids

