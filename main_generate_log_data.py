import argparse
from Config.generate_log_data_config import config
from Controller.Controller import Controller
from Common.logger import create_log_and_device
from Common.Utils import set_global_seed
from DataLogger.datalogger import DataLogger
import numpy as np
from Agent.ControlAgent import ControlAgents


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="control_agent")
    parser.add_argument("--task", type=str, default="bcb")
    parser.add_argument("--nebula-mdl", action='store_true')
    parser.add_argument("--tables", type=str, default="local_tables")
    return parser.parse_args()


def run_train(args=get_args()):
    logger, device = create_log_and_device(args, config, algo_name="nash_bcb")
    set_global_seed(config["seed"])

    data_logger = DataLogger(config["basic_config"]["num_agents"])

    # init the budget and cate
    # determine the budget and the cate
    budgets = np.random.choice(config["basic_config"]["budget_level"], config["basic_config"]["num_agents"],
                               replace=True, )
    cate = np.arange(config["basic_config"]["num_agents"]) // config["basic_config"]["agent_cate"]

    control_agents = ControlAgents(budgets=budgets[:config["basic_config"]["num_controllable_agent"]],
                                   name="ControlAgents",
                                   num_agents=config["basic_config"]["num_controllable_agent"],
                                   budget_max=max(config["basic_config"]["budget_level"]),
                                   agent_cate_list=cate[:config["basic_config"]["num_controllable_agent"]],
                                   num_tick=config["basic_config"]["num_tick"],
                                   pv_num=config["basic_config"]["pv_num"],
                                   deterministic=True,
                                   device=device,
                                   obs_dim=config["ControlAgent"]["obs_dim"],
                                   cate_id_dim=config["ControlAgent"]["id_cate_dim"],
                                   embedding_dim=config["ControlAgent"]["embedding_dim"],
                                   embedding_hidden_layers=config["ControlAgent"]["embedding_hidden_dims"],
                                   common_hidden_dim=config["ControlAgent"]["base_actor_hidden_dims"],
                                   id_onehot_output_act_fn=config["ControlAgent"]["embedding_out_act_fn"],
                                   actor_bb_out_dim=config["ControlAgent"]["actor_bb_out_dim"],
                                   action_dim=config["ControlAgent"]["action_dim"]
                                   )

    controller = Controller(num_controllable_agent=config["basic_config"]["num_controllable_agent"],
                            num_uncontrollable_agents=config["basic_config"]["num_uncontrollable_agents"],
                            budget_level=config["basic_config"]["budget_level"],
                            num_agent_cate=config["basic_config"]["agent_cate"],
                            num_tick=config["basic_config"]["num_tick"],
                            reserve_pv_price=config["basic_config"]["reserve_pv_price"],
                            min_remaining_budget=config["basic_config"]["min_remaining_budget"],
                            is_sparse=config["basic_config"]["is_sparse"],
                            pv_num=config["basic_config"]["pv_num"],
                            deterministic=config["ControlAgent"]["deterministic"],
                            device=device,
                            data_logger=data_logger,
                            control_agents=control_agents,
                            budgets=budgets,
                            cate=cate)

    _, datalogger = controller.run_simulation(num_trajectory=config["generate_log_params"]["simulate_episode"],
                                              save_data=True)
    datalogger.save()


if __name__ == '__main__':
    run_train()
