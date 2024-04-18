from DataLogger.datalogger import DataLogger
from Config.config import config
from Algorithm.PE_MRL import PEMRL
import numpy as np
import torch


def run():
    # init seeds
    np.random.seed(1)
    torch.manual_seed(1)

    # load data
    datalogger = DataLogger(config["basic_config"]["num_agents"])
    datalogger.load()
    env_train_data = datalogger.generate_train_data()

    # init PE-MRL
    pemrl = PEMRL(config["TransitionModel"], config["SAC"], datalogger.data, env_train_data,
                  config["basic_config"]["budgets"])

    # train
    pemrl.train()


if __name__ == '__main__':
    run()
