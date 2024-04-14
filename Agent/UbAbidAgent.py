import numpy as np
from Agent.BaseAgent import BaseAgent


class UbAbidAgent(BaseAgent):
    def __init__(self, budget=100, name="UbAbidAgent", category=0, roi=1,
                 num_tick=24, bid_scale=0.1):
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

    def action(self, tickIndex, pv_pvalues,):
        alpha = self.base_actions[tickIndex] / pv_pvalues.mean()
        bids = alpha * pv_pvalues
        return bids, alpha


