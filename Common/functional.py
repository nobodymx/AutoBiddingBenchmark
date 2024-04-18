import math
import torch
import numpy as np
from copy import deepcopy
import os


def minibatch_inference(batch, rollout_fn, batch_size=1000, cat_dim=0, device='cpu'):
    if not isinstance(batch, torch.Tensor):
        batch = torch.Tensor(batch).to(device)
    data_size = batch.shape[0]
    num_batches = int(np.ceil(data_size / batch_size))
    inference_results = []
    multi_op = False
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min(data_size, (i + 1) * batch_size)
        input_batch = batch[batch_start:batch_end]
        outputs = rollout_fn(input_batch)
        if i == 0:
            if isinstance(outputs, tuple):
                multi_op = True
            else:
                multi_op = False
            inference_results = outputs
        else:
            if multi_op:
                inference_results = (torch.cat([prev_re, op], dim=cat_dim) for prev_re, op in
                                     zip(inference_results, outputs))
            else:
                inference_results = torch.cat([inference_results, outputs])
    return inference_results


def termination_fn(next_time_left):
    res = np.zeros(len(next_time_left))
    for i in range(len(next_time_left)):
        res[i] = 1 if next_time_left[i] <= -0.967 else 0
    return res


def termination_fn2(next_time_left, next_budget_left):
    res = 1 if next_time_left <= -0.967 or next_budget_left <= -0.998 else 0
    return res


def dict_batch_generator(data, batch_size, keys=None):
    if keys is None:
        keys = list(data.keys())
    num_data = len(data[keys[0]])
    num_batches = int(np.ceil(num_data / batch_size))
    indices = np.arange(num_data)
    np.random.shuffle(indices)
    for batch_id in range(num_batches):
        batch_start = batch_id * batch_size
        batch_end = min(num_data, (batch_id + 1) * batch_size)
        batch_data = {}
        for key in keys:
            batch_data[key] = data[key][indices[batch_start:batch_end]]
        yield batch_data


def change_std_data_into_env_model_roll_out_data(std_data, input_data_flag=True, ground_truth_flag=True,
                                                 ):
    """
    :param ground_truth_flag:
    :param input_data_flag:
    :param std_data: {"observations":,"actions":,"next_observations":,"rewards":,"terminals":, "tick_cost_ratios":}
    :return:
    """
    input_data, ground_truth = None, None
    if input_data_flag:
        input_data = np.concatenate(
            [std_data["observations"], std_data["actions"],
             np.expand_dims(std_data["next_time"], axis=1)],
            axis=1)
    if ground_truth_flag:
        ground_truth = np.concatenate(
            [np.expand_dims(-std_data["next_observations"][:, 1] + std_data["observations"][:, 1],
                            axis=1),
             np.expand_dims(std_data["next_observations"][:, 3] - std_data["observations"][:, 3],
                            axis=1),
             std_data["rewards"],
             std_data["tick_cost_ratios"]],
            axis=1)
    return input_data, ground_truth


def change_std_data_into_env_model_roll_out_id_fea_data(std_data, input_data_flag=True, ground_truth_flag=True,
                                                        ):
    """
    :param ground_truth_flag:
    :param input_data_flag:
    :param std_data: {"observations":,"actions":,"next_observations":,"rewards":,"terminals":, "tick_cost_ratios":}
    :return:
    """
    input_data, ground_truth = None, None
    if input_data_flag:
        """
        observations, actions, new_ratio, id_feature, budget, next_time, 
        """
        input_data = np.concatenate(
            [std_data["observations"],
             std_data["actions"],
             std_data["new_ratio"],
             std_data["budget"],
             np.expand_dims(std_data["next_time"], axis=1),
             std_data["id_feature"]],
            axis=1)

    if ground_truth_flag:
        """
        delta budget left, delta paoa, reward, pcost
        """
        ground_truth = np.concatenate(
            [np.expand_dims(-std_data["next_observations"][:, 1] + std_data["observations"][:, 1],
                            axis=1),
             np.expand_dims(std_data["next_observations"][:, 3] - std_data["observations"][:, 3],
                            axis=1),
             std_data["rewards"],
             std_data["tick_cost_ratios"]],
            axis=1)
    return input_data, ground_truth


def change_env_model_pred_data_into_std_data(pred_output, env_input, next_now_ratio, next_rest_ratio, next_time_left,
                                             action, if_fea=False, fea_dim=100,
                                             ):
    """
    :param fea_dim:
    :param if_fea:
    :param action:
    :param next_time_left:
    :param env_input:
    :param next_rest_ratio:
    :param next_now_ratio:
    :param pred_output:
    :return:
    """
    if isinstance(env_input, torch.Tensor):
        env_input = env_input.detach().cpu().numpy()
    tick_cost_ratio = pred_output[:, 3]
    tick_cost_ratio = np.max(
        np.concatenate([np.expand_dims(tick_cost_ratio, axis=1), np.zeros((len(tick_cost_ratio), 1))], axis=1),
        axis=1)  # not smaller than zero
    reward = pred_output[:, 2]
    delta_b_t = pred_output[:, 0]
    b_t_1 = env_input[:, 1] - delta_b_t

    b_t_1_real = np.max(np.concatenate([np.expand_dims((b_t_1 + 1) / 2, axis=1), np.zeros((len(b_t_1), 1))], axis=1),
                        axis=1) + 1e-6  # b_t_1_real >= 0

    # calculate the spend speed
    try:
        ss = np.expand_dims(tick_cost_ratio, axis=1) * next_rest_ratio / np.expand_dims(b_t_1_real,
                                                                                        axis=1) / next_now_ratio - 1
        ss = np.min(np.concatenate([ss, np.ones((len(tick_cost_ratio), 1))], axis=1), axis=1)
    except:
        print("calculate ss wrong")
        assert 0
    if True in np.isnan(ss):
        # modified the nan value: this nan value is caused by the zero value of next_now_ratio
        lst = np.isnan(ss)
        ss[lst] = tick_cost_ratio[lst] / b_t_1_real[lst] - 1
        ss[lst] = np.min(
            np.concatenate([np.expand_dims(ss[lst], axis=1), np.ones((len(tick_cost_ratio[lst]), 1))], axis=1), axis=1)
    if True in np.isnan(ss):
        print("spend_speed cal is None")
        for i in range(len(ss)):
            print("ss")
            print(ss[i])
            print("tick_cost_ratio")
            print(tick_cost_ratio[i])
            print("next_rest_ratio")
            print(next_rest_ratio[i])
            print("next_now_ratio")
            print(next_now_ratio[i])
            print("b_t_1_real")
            print(b_t_1_real[i])
        # assert 0

    next_paoa = pred_output[:, 1] + env_input[:, 3]
    if if_fea:
        next_obs = np.concatenate(
            [np.expand_dims(next_time_left, axis=1), np.expand_dims(b_t_1, axis=1), np.expand_dims(ss, axis=1),
             np.expand_dims(next_paoa, axis=1),
             np.expand_dims(env_input[:, 4], axis=1),
             action,
             np.expand_dims(env_input[:, 2], axis=1),
             env_input[:, 7:11],
             env_input[:, -fea_dim - 2:-2]
             ],
            axis=1)
    else:
        next_obs = np.concatenate(
            [np.expand_dims(next_time_left, axis=1), np.expand_dims(b_t_1, axis=1), np.expand_dims(ss, axis=1),
             np.expand_dims(next_paoa, axis=1),
             np.expand_dims(env_input[:, 4], axis=1),
             action,
             np.expand_dims(env_input[:, 2], axis=1),
             env_input[:, 7:11]
             ],
            axis=1)
    return next_obs, reward


def shuffle(data):
    """
    :param data:{key:data}
    :return:
    """
    size = 0
    for key in data:
        size = len(data[key])
        break
    ind = np.arange(size)
    np.random.shuffle(ind)
    return {
        key: data[key][ind].copy() for key in data
    }


def cal_ts_from_un_normal_tl(tl):
    return int(math.floor((1 - tl) / 0.020833))


def cal_ratios(traj_time_step, traj_now_ratio, traj_rest_ratio, tls):
    """
    :param traj_time_step:
    :param traj_now_ratio:
    :param traj_rest_ratio:
    :param tls: next time left set
    :return:
    """
    traj_now_ratio = np.array(traj_now_ratio)
    traj_rest_ratio = np.array(traj_rest_ratio)
    if len(tls) + 1 == len(traj_time_step):
        traj_now_ratio = traj_now_ratio[np.argsort(traj_time_step)]
        traj_rest_ratio = traj_rest_ratio[np.argsort(traj_time_step)]
        return np.expand_dims(traj_now_ratio[1:], 1), np.expand_dims(traj_rest_ratio[1:], 1)
    else:
        res_now_ratio = np.zeros((len(tls), 1))
        res_rest_ratio = np.zeros((len(tls), 1))
        for j in range(len(tls)):
            t = cal_ts_from_un_normal_tl(tls[j])  # next time step
            try:
                res_now_ratio[j] = traj_now_ratio[traj_time_step.index(t)]
                res_rest_ratio[j] = traj_rest_ratio[traj_time_step.index(t)]
            except:
                opt_index, min_dist = -1, 1e10
                for i in range(len(traj_time_step)):
                    if abs(traj_time_step[i] - t) < min_dist:
                        min_dist = abs(traj_time_step[i] - t)
                        opt_index = i
                res_now_ratio[j] = traj_now_ratio[opt_index]
                res_rest_ratio[j] = traj_rest_ratio[opt_index]
        return res_now_ratio, res_rest_ratio


def cal_instance_ratio(tls, now_ratio, rest_ratio, time_step):
    length = len(tls)
    next_now_ratio = np.zeros([length, 1])
    next_rest_ratio = np.zeros([length, 1])
    for i in range(length):
        t = cal_ts_from_un_normal_tl((tls[i] + 1) / 2)  # next time step
        try:
            next_now_ratio[i] = now_ratio[i][time_step[i].tolist().index(t)]
            next_rest_ratio[i] = rest_ratio[i][time_step[i].tolist().index(t)]
        except:
            opt_index, min_dist = -1, 1e10
            for j in range(len(time_step[i])):
                if abs(time_step[i][j] - t) < min_dist:
                    min_dist = abs(time_step[i][j] - t)
                    opt_index = j
            next_now_ratio[i] = now_ratio[i][opt_index]
            next_rest_ratio[i] = rest_ratio[i][opt_index]
    return next_now_ratio, next_rest_ratio


def find_now_rest_ratio(tick_index, now_ratio_lst, rest_ratio_lst, time_step):
    try:
        now_ratio = now_ratio_lst[time_step.tolist().index(tick_index)]
        rest_ratio = rest_ratio_lst[time_step.tolist().index(tick_index)]
    except:
        opt_index, min_dist = -1, 1e10
        for j in range(len(time_step)):
            if abs(time_step[j] - tick_index) < min_dist:
                min_dist = abs(time_step[j] - tick_index)
                opt_index = j
        now_ratio = now_ratio_lst[opt_index]
        rest_ratio = rest_ratio_lst[opt_index]
    return now_ratio, rest_ratio


def change_std_data_into_multi_agent_env_data(state, id_feature, action, reward, next_state, tick_cost_ratio):
    inputs = np.concatenate([
        state, id_feature, np.array([action]), np.array([next_state[0]])
    ])
    ground_truths = np.array([
        state[1] - next_state[1],
        next_state[3] - state[3],
        reward,
        tick_cost_ratio
    ]
    )
    return inputs, ground_truths


def change_input_to_input_with_id(input_data, cate_level, cate_level_num):
    num = len(input_data)
    input_cate_data = [np.zeros((num, cate_level_num[key])) for key in cate_level_num]
    input_data_ = [deepcopy(input_data[:, :-cate_level])]
    for level in range(cate_level):
        for index in range(num):
            input_cate_data[level][index, int(input_data[index, -cate_level + level])] = 1
        input_data_.append(deepcopy(input_cate_data[level]))
    input_data = np.concatenate(input_data_, 1)
    return input_data


def init_random_cate_level(cate_level, cate_level_num):
    cate_info = []
    for i in range(1, cate_level + 1):
        temp_cate = np.zeros(cate_level_num[i])
        temp_cate[np.random.randint(cate_level_num[i])] = 1
        cate_info.append(deepcopy(temp_cate))
    cate_info = np.concatenate(cate_info)
    return cate_info


def change_std_data_into_policy_input_with_id(observation, id_fea, cate_level, cate_level_num):
    num = len(observation)
    input_data = [observation]
    input_data.extend([np.zeros((num, cate_level_num[key])) for key in cate_level_num])
    for level in range(cate_level):
        for index in range(num):
            input_data[level + 1][index, id_fea[index, level]] = 1
    input_data = np.concatenate(input_data, 1)
    return input_data


def change_org_ratio_to_new_ratio(org_ratio, actions):
    return org_ratio * (1 + actions)


def shuffle_and_padding(data):
    length = len(data)
    x, y, ground_truth = [], [], []
    max_N_1 = 0
    input_dim = len(data[0]["inputs"][0])
    for i in range(length):
        x.append(torch.Tensor(data[i]["inputs"]))
        temp_x = torch.Tensor(data[i]["inputs"])
        N = len(temp_x)
        max_N_1 = N - 1 if N - 1 > max_N_1 else max_N_1
        masks = torch.ones(N, N)
        diag = torch.diag(masks)
        mask_diag = torch.diag_embed(diag)
        masks -= mask_diag
        masks = masks.type(torch.bool)
        y.append(torch.stack(
            [torch.masked_select(temp_x.transpose(1, 0), masks[i]).view(input_dim, N - 1).transpose(1, 0) for i in
             range(N)]))
        ground_truth.append(torch.Tensor(data[i]["ground_truth"]))
    x = torch.cat(x, 0)
    ground_truth = torch.cat(ground_truth, 0)
    for i in range(len(y)):
        now_N_1 = y[i].shape[1]
        zeros = torch.zeros(now_N_1 + 1, max_N_1 - now_N_1, input_dim)
        y[i] = torch.cat([y[i], zeros], 1)
    y = torch.cat(y, 0)

    # shuffle
    size = len(y)
    ind = np.arange(size)
    np.random.shuffle(ind)
    x = x[ind]
    y = y[ind]
    ground_truth = ground_truth[ind]

    return x, y, ground_truth


def make_other_agents_states(states):
    input_dim = states.shape[1]
    N = states.shape[0]
    masks = torch.ones(N, N)
    diag = torch.diag(masks)
    mask_diag = torch.diag_embed(diag)
    masks -= mask_diag
    masks = masks.type(torch.bool)
    y = torch.stack(
        [torch.masked_select(torch.Tensor(states).transpose(1, 0), masks[i]).view(input_dim, N - 1).transpose(1, 0) for
         i in
         range(N)])
    return y


def shuffle_torch(x, y, ground_truth):
    # shuffle
    size = len(y)
    ind = np.arange(size)
    np.random.shuffle(ind)
    x = x[ind]
    y = y[ind]
    ground_truth = ground_truth[ind]
    return x, y, ground_truth


def log_prob(action, mu, sigma):
    var = (sigma ** 2)
    log_scale = sigma.log()
    log_probability = -((action - mu) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    return log_probability


def save_agents_info(store_agent_data, logger, path='info/agents_info.txt'):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("Agents Info:\n")
        for i in range(len(store_agent_data)):
            f.write("Agent " + str(i) + '\n')
            f.write(store_agent_data[i]["key"] + '\n')
            f.write(store_agent_data[i]["budget_level"] + '\n')
            f.write(store_agent_data[i]["budget_level_info"] + '\n')
            f.write(store_agent_data[i]["cate_feature"] + '\n')
            f.write(store_agent_data[i]["now_ratio_seq"] + '\n')
            f.write(store_agent_data[i]["rest_ratio_seq"] + '\n')
            f.write(store_agent_data[i]["time_step"] + '\n')
    save_dir = os.path.join(logger.log_path, 'agents_info')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "agents_info.txt")
    os.system("cp " + path + " " + save_path)
    logger.info_print("Save agents info")
    assert 0
