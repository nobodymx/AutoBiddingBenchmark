import numpy as np


class BaseAgent:
    def __init__(self, budget=100, name="RuleAgent", category=0):
        self.budget = budget
        self.remaining_budget = budget
        self.name = name
        self.category = category

    def reset(self):
        pass
