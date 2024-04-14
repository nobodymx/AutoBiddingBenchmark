import numpy as np
from Agent.BaseAgent import BaseAgent


class RuleAgent(BaseAgent):
    def __init__(self, budget=100, name="RuleAgent", category=0, roi=2):
        super().__init__()
        self.budget = budget
        self.remaining_budget = budget
        self.base_actions = np.array(
            [22.1, 28.7, 27.8, 25.6, 20.8, 18.6, 18.6, 18.3, 19.6, 19.5, 18.7, 19.5, 22.5, 23.0, 23.5, 22.4,
             22.7, 24.7, 27.6, 31.6, 30.8, 25.6, 23.7, 21.4])
        self.name = name
        self.category = category
        self.roi = roi

    def reset(self):
        self.remaining_budget = self.budget

    def action(self, tickIndex, pv_pvalues, ):
        shape_of_pv_values = pv_pvalues.shape
        bids = np.full(shape_of_pv_values, self.base_actions[tickIndex])
        bids = bids + np.random.random(bids.shape)
        return bids



