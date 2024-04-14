config = {
    "seed": 0,
    "gpu_enable": True,
    "gpu_id": "0",
    "logdir": "./log",
    "basic_config": {
        "num_controllable_agent": 1,
        "num_uncontrollable_agents": 29,
        "num_agents": 30,
        "budget_level": [50, 100, 200, 500, 1000, 2000],
        "agent_cate": 6,
        "num_tick": 24,
        "reserve_pv_price": 0.0001,
        "min_remaining_budget": 0.1,
        "is_sparse": False,
        "device": "cpu",
        "pv_num": 7700000,
    },

    "generate_log_params": {
        "simulate_episode": 10
    },


    "ControlAgent": {
        "obs_dim": 4,
        "action_dim": 1,
        "id_cate_dim": 7,
        "embedding_out_act_fn": "tanh",
        "actor_bb_out_dim":10,
        "base_actor_hidden_dims": [50, 50],
        "base_critic_hidden_dims": [100, 100],
        "embedding_hidden_dims": [200, 200],
        "embedding_dim": 3,
        "act_fn": "relu",
        "out_act_fn": "identity",
        "actor_lr": 2e-5,
        "critic_lr": 2e-5,
        "gamma": 1,
        "lr_decay": 1.0,
        "train_epoch": 10000,
        "num_trajectory": 10,
        "tau": 0.1,
        "deterministic": True,
        "step": 10,
    },
}
