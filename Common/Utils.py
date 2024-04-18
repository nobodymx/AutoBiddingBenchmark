import math
import numpy as np
import random
import torch


def log_prob(action, mu, sigma):
    var = (sigma ** 2)
    log_scale = sigma.log()
    log_probability = -((action - mu) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    return log_probability


def calibration_by_remaining_budget(reward, remaining_budget, cost, over_cost_ratio):
    calibrated_reward = reward * (1 - over_cost_ratio)
    calibrated_cost = np.minimum(cost, remaining_budget)
    return calibrated_reward, calibrated_cost


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_moniters(moniter, pi_name="coop", logger=None, num_controllable_agents=10, ):
    assert logger is not None
    if pi_name == "coop/theta" or pi_name == "inner_opt/pi_nu_star":
        average_reward = 0
        average_cost = 0
        for i in range(num_controllable_agents):
            logger.info_print(
                "[eval/" + pi_name + "]: agent " + str(i + 1) + " reward: " + str(np.mean(moniter[i]["rewards"])))
            logger.info_print(
                "[eval/" + pi_name + "]: agent " + str(i + 1) + " cost: " + str(np.mean(moniter[i]["costs"])))
            average_reward += np.mean(moniter[i]["rewards"])
            average_cost += np.mean(moniter[i]["costs"])
        average_reward /= num_controllable_agents
        average_cost /= num_controllable_agents
        logger.info_print("[eval/" + pi_name + "]: average reward: " + str(average_reward))
        logger.info_print("[eval/" + pi_name + "]: average cost: " + str(average_cost))

    elif pi_name == "comp" or pi_name == "comp_star":
        logger.info_print(
            "[eval/" + pi_name + "/rep_nu]: reward: " + str(np.mean(moniter["rep"]["rewards"])))
        logger.info_print(
            "[eval/" + pi_name + "/rep_nu]: cost: " + str(np.mean(moniter["rep"]["costs"])))
        logger.info_print(
            "[eval/" + pi_name + "/back_theta]: reward: " + str(np.mean(moniter["back"]["rewards"])))
        logger.info_print(
            "[eval/" + pi_name + "/back_theta]: cost: " + str(np.mean(moniter["back"]["costs"])))


def logger_record(monitor, pi_name="coop", step=100, logger=None, num_agents=10):
    if pi_name == "theta":
        average_reward = 0
        average_cost = 0
        for i in range(num_agents):
            logger.record(
                "[eval/" + pi_name + "]: agent " + str(i + 1) + " reward: ", np.mean(monitor[i]["rewards"]),
                step)
            logger.record(
                "[eval/" + pi_name + "]: agent " + str(i + 1) + " cost: ", np.mean(monitor[i]["costs"]), step)
            average_reward += np.mean(monitor[i]["rewards"])
            average_cost += np.mean(monitor[i]["costs"])
        average_reward /= num_agents
        average_cost /= num_agents
        logger.record("[eval/" + pi_name + "]: average reward: ", average_reward, step)
        logger.record("[eval/" + pi_name + "]: average cost: ", average_cost, step)
