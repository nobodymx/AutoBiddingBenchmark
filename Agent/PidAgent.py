import numpy as np
from Agent.BaseAgent import BaseAgent


class PidAgent(BaseAgent):
    def __init__(self, budget=100, name="PidAgent", category=0, roi=1):
        super().__init__()
        self.budget = budget
        self.remaining_budget = budget
        self.base_actions = 0.05
        self.name = name
        self.category = category
        self.exp_budget_ratio = np.arange(1 / 24, 1, 1 / 24)
        self.alpha = None
        self.last_remaining_budget = self.remaining_budget
        self.roi = roi

    def reset(self):
        self.remaining_budget = self.budget

    def action(self, tickIndex, pv_pvalues):

        if tickIndex == 0:
            self.alpha = self.base_actions / pv_pvalues.mean()
        else:
            last_tick_cost = self.last_remaining_budget - self.remaining_budget
            self.last_remaining_budget -= last_tick_cost
            if last_tick_cost * (24 - tickIndex) / self.remaining_budget < 0.8:
                self.alpha *= 1.13
            elif last_tick_cost * (24 - tickIndex) / self.remaining_budget > 1.2:
                self.alpha *= 0.8
        bids = self.alpha * pv_pvalues
        return bids, self.alpha

    def update(self, cost):
        self.remaining_budget -= cost

