import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class MultiNormalPvGen:
    def __init__(self, num_tick=24, num_agent_category=6, num_caterory=5, pv_num=10000,
                 num_agents=30):
        super().__init__()
        self.num_tick = num_tick
        self.num_agent_category = num_agent_category
        self.num_caterory = num_caterory
        self.num_agent = num_agents
        self.pv_num = pv_num
        cvr_R_matrix = torch.tensor([
            [1, -0.9, 0.5, -0.4, 0.2, 0.4],
            [-0.9, 1, -0.4, 0.1, -0.3, -0.4],
            [0.5, -0.4, 1, -0.6, 0.6, 0.5],
            [-0.4, 0.1, -0.6, 1, -0.5, -0.1],
            [0.2, -0.3, 0.6, -0.5, 1, 0.3],
            [0.4, -0.4, 0.5, -0.1, 0.3, 1],
        ])  # 前4维代表类目，最后一维代表时间，彼此之间存在一定的相关性

        # 定义方差对角阵
        cvr_var = torch.tensor([0.001, 0.001, 0.004, 0.009, 0.0001, 500])
        D_matrix = torch.diag(cvr_var)
        # 计算协方差矩阵
        self.cvr_cov_matrix = D_matrix.sqrt().mm(cvr_R_matrix).mm(D_matrix.sqrt())
        # 定义各维度均值
        self.mean = torch.tensor([0.05, 0.03, 0.07, 0.06, 0.01, 48])

        self.pv_values, self.pv_categories = self.generate()

    def reset(self):
        # self.pv_values, self.pv_categories = self.generate()
        pass

    def generate(self):
        # 生成多元高斯分布
        multivariate_normal_cvr = MultivariateNormal(self.mean, self.cvr_cov_matrix)
        # 生成样本（类目）
        samples = multivariate_normal_cvr.sample((self.pv_num,))

        # 流量的时间戳
        self.time_stamps = samples[:, -1].clamp(0, 96) / (96 / self.num_tick)

        # ctr 暂时默认为1，如果需要的话可以复杂化。
        ctr = torch.ones_like(samples[:, :-1])
        self.pv_matrix_category = samples[:, :-1] * ctr
        # 流量智能体矩阵
        pv_matrix_agent = self.pv_matrix_category.repeat_interleave(repeats=self.num_agent_category, dim=1)
        pv_matrix_agent = pv_matrix_agent + self.gaussian_noise(0, 0.01, pv_matrix_agent.shape)
        # cvr默认在0～1之间
        pv_matrix_agent = pv_matrix_agent.clamp(0, 10)
        pv_values, pv_categorys = list(), list()
        for i in range(self.num_tick):
            # 选取合适的时间范围
            tick_pv = pv_matrix_agent[(self.time_stamps >= i) & (self.time_stamps < i + 1)]
            pv_values.append(tick_pv.numpy())
            pv_categorys.append(torch.zeros(tick_pv.shape[0], 1).numpy())

        return pv_values, pv_categorys

    def gaussian_noise(self, mean, stddev, shape):
        gaussian_noise = torch.randn(shape)
        return mean + stddev * gaussian_noise

    def draw_density(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        # 绘制密度热力图
        _, axes = plt.subplots(4, 4, figsize=(10, 8))
        _, axes2 = plt.subplots(3, 2, figsize=(10, 8))

        for i in range(4):
            for j in range(i + 1, 5):
                sns.kdeplot(x=self.pv_matrix_category[:1000, i], y=self.pv_matrix_category[:1000, j], cmap='PuBu',
                            fill=True, thresh=0.05,
                            ax=axes[i, j - 1])
        axis_index = 0
        for i in range(3):
            for j in range(2):
                if axis_index < 5:
                    print(i, j)
                    sns.kdeplot(x=self.time_stamps[:1000], y=self.pv_matrix_category[:1000, axis_index], cmap='PuBu',
                                fill=True,
                                thresh=0.05, ax=axes2[i, j])
                    axis_index += 1
        plt.show()


if __name__ == '__main__':
    pvGen = MultiNormalPvGen()
    pvGen.reset()
    print(" ")
