import numpy as np
from Agent.BaseAgent import BaseAgent


class RAgent(BaseAgent):
    def __init__(self, budget=100, name="RAgent", category=0, roi=1.0):
        super().__init__()
        self.budget = budget
        self.remaining_budget = budget
        self.base_actions = 0.07
        self.name = name
        self.category = category
        self.exp_budget_ratio = np.arange(1 / 24, 1, 1 / 24)
        self.remaining_budget = budget
        self.alpha = None

    def reset(self):
        self.remaining_budget = self.budget

    def action(self, tickIndex, pv_pvalues=0,):
        if tickIndex == 0:
            self.alpha = np.random.randint(0, 2)
        else:
            if (self.budget - self.remaining_budget) / self.budget < self.exp_budget_ratio[tickIndex] * 0.8:
                self.alpha *= 1.12
            elif (self.budget - self.remaining_budget) / self.budget > self.exp_budget_ratio[tickIndex] * 1.2:
                self.alpha /= 0.8
        bids = self.alpha * pv_pvalues
        return bids, self.alpha

    def update(self, cost):
        self.remaining_budget -= cost



