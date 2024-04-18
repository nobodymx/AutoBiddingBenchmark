import matplotlib.pyplot as plt


def draw_rewards(rewards):
    x = [i for i in range(len(rewards))]
    plt.plot(x, rewards)
    plt.savefig("Results/rewards.png")
    plt.close()
