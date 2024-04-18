import torch
import torch.nn as nn
import os
from Models.ensemble_model_PE_model import PEEnsembleModel
from torch.optim.lr_scheduler import CosineAnnealingLR as CALR
import numpy as np
from copy import deepcopy
from Common.functional import dict_batch_generator, change_env_model_pred_data_into_std_data, \
    termination_fn2


class TransitionModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=10, ensemble_size=4, num_elite=1,
                 decay_weights=None, hidden_layer_num=10, hidden_io=100, embedding_dim=100,
                 head_num=8, v_hidden_layers=10, head_extraction_layers=5, trans_output_dim=500,
                 lr=0.01, kq_dim=100, mlp_hidden_layer=None,
                 trans_mlp_hidden_layers=None,
                 total_mlp_hidden_layers=None,
                 use_weight_decay=False,
                 act_fn="swish", out_act_fn="identity",
                 device="cpu", logger=None, elite_index="mae",
                 eval_percent=0.2,
                 eval_batch_size=10,
                 training_epoch=10000,
                 lr_decay=1.0,
                 if_id_cate=True,
                 id_cate_embedding=100,
                 reward_penalty_coeff=0.1,
                 inference_batch_max=100,
                 obs_dim=40,
                 beta=5):
        super(TransitionModel, self).__init__()

        self.logger = logger
        self.elite_index = elite_index
        self.eval_percent = eval_percent
        self.inference_batch_max = inference_batch_max
        self.lr = lr
        self.obs_dim = obs_dim
        self.use_weight_decay = use_weight_decay
        self.env_model_batch_size = eval_batch_size
        self.if_id_cate = if_id_cate
        self.reward_penalty_coeff = reward_penalty_coeff
        self.beta = beta

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mean_dim = int(output_dim / 2)
        self.var_dim = int(output_dim / 2)

        self.id_cate_embedding = id_cate_embedding

        self.device = device
        self.ensemble_size = ensemble_size
        self.num_elite = num_elite
        self.decay_weights = decay_weights

        self.env_model = PEEnsembleModel(input_dim, output_dim, hidden_layer_num, hidden_io, trans_output_dim,
                                         embedding_dim,
                                         head_num, v_hidden_layers, head_extraction_layers, kq_dim, mlp_hidden_layer,
                                         ensemble_size=ensemble_size,
                                         num_elite=num_elite, decay_weights=decay_weights,
                                         trans_mlp_hidden_layers=trans_mlp_hidden_layers,
                                         total_mlp_hidden_layers=total_mlp_hidden_layers,
                                         act_fn=act_fn, out_act_fn=out_act_fn, device=self.device)

        self.model_optimizer = torch.optim.Adam(self.env_model.parameters(), self.lr)
        self.sch = CALR(self.model_optimizer, training_epoch, eta_min=self.lr / lr_decay)

        # reset the best snapshot, define in the init function
        self.model_best_snapshots = [deepcopy(self.env_model.ensemble_models[idx].state_dict()) for idx in
                                     range(self.ensemble_size)]
        self.best_snapshot_losses = [1e10 for _ in range(self.ensemble_size)]
        self.best_snapshot_losses_dim_lst = np.ones((self.ensemble_size, self.mean_dim)) * 1e10

        self.to(device)

    @torch.no_grad()
    def eval(self, eval_data, update_elite_models=False):
        """
        :param self:
        :param eval_data:
        :param update_elite_models:
        :return:
        """

        x, y, ground_truth = eval_data[0], eval_data[1], eval_data[2]
        # x, y, ground_truth = shuffle_torch(x, y, ground_truth)

        x = x
        y = y
        ground_truth = ground_truth

        eval_mse_losses, eval_mse_losses_every_dim, eval_mape_losses, eval_mape_losses_every_dim, eval_mae_losses, eval_mae_loss_every_dim = [], [], [], [], [], []
        for eval_batch in dict_batch_generator({"x": x,
                                                "y": y,
                                                "ground_truth": ground_truth},
                                               self.env_model_batch_size):
            temp_x, temp_y, temp_gt = eval_batch["x"].to(self.device), eval_batch["y"].to(self.device), eval_batch[
                "ground_truth"].to(self.device)
            predictions = self.predict(temp_x, temp_y)
            temp_eval_mse_losses, temp_eval_mse_losses_every_dim, temp_eval_mape_losses, temp_eval_mape_losses_every_dim, temp_eval_mae_losses, temp_eval_mae_loss_every_dim = self.model_loss(
                predictions, temp_gt, type_="eval")
            eval_mse_losses.append(temp_eval_mse_losses)
            eval_mse_losses_every_dim.append(temp_eval_mse_losses_every_dim)
            eval_mape_losses.append(temp_eval_mape_losses)
            eval_mape_losses_every_dim.append(temp_eval_mape_losses_every_dim)
            eval_mae_losses.append(temp_eval_mae_losses)
            eval_mae_loss_every_dim.append(temp_eval_mae_loss_every_dim)

        eval_mse_losses = torch.mean(torch.stack(eval_mse_losses), dim=0)
        eval_mse_losses_every_dim = torch.mean(torch.stack(eval_mse_losses_every_dim), dim=0)
        eval_mape_losses = torch.mean(torch.stack(eval_mape_losses), dim=0)
        eval_mape_losses_every_dim = torch.mean(torch.stack(eval_mape_losses_every_dim), dim=0)
        eval_mae_losses = torch.mean(torch.stack(eval_mae_losses), dim=0)
        eval_mae_loss_every_dim = torch.mean(torch.stack(eval_mae_loss_every_dim), dim=0)
        if update_elite_models:
            self.logger("Update the elite env model index")
            if self.elite_index == "mape":
                elite_idx = list(np.argsort(eval_mape_losses))
            elif self.elite_index == "mse":
                elite_idx = list(np.argsort(eval_mse_losses))
            else:
                elite_idx = list(np.argsort(eval_mae_losses))
            self.model.elite_model_idxes = torch.tensor(elite_idx[:self.env_model.num_elite])
        return eval_mse_losses, eval_mse_losses_every_dim, eval_mape_losses, eval_mape_losses_every_dim, eval_mae_losses, eval_mae_loss_every_dim

    def model_loss(self, predictions, ground_truth, type_="train"):
        pred_mean, pred_logvar = predictions
        print(pred_mean[0])
        if type_ == "train":
            inv_var = torch.exp(-pred_logvar)
            mse_losses = torch.mean(torch.mean(torch.pow(pred_mean - ground_truth, 2) * inv_var, dim=(2)), dim=1)
            var_losses = torch.mean(torch.mean(pred_logvar, dim=(2)), dim=1)
            return mse_losses, var_losses
        elif type_ == "eval":
            mse_losses = torch.mean(torch.pow(pred_mean - ground_truth, 2), dim=(1, 2,))
            # calculate the mse over all data, return [num_model, pred_dim]
            mse_losses_every_dim = torch.mean(torch.pow(pred_mean - ground_truth, 2), dim=(1,))
            # MAPE
            mape_losses = torch.mean(torch.abs((pred_mean - ground_truth) / (ground_truth + 1e-4)), dim=(1, 2))
            mape_losses_every_dim = torch.mean(torch.abs((pred_mean - ground_truth) / (ground_truth + 1e-4)),
                                               dim=(1))

            # MAE
            mae_loss = torch.mean(torch.abs((pred_mean - ground_truth)), dim=(1, 2))
            mae_loss_every_dim = torch.mean(torch.abs((pred_mean - ground_truth)), dim=(1))

            return mse_losses, mse_losses_every_dim, mape_losses, mape_losses_every_dim, mae_loss, mae_loss_every_dim
        else:
            assert 0

    def predict(self, x, y):
        """
        :param inputs: [batch, channel, feature]
        :return: [N, batch, channel, mean_dim], [N, batch, channel, var_dim]
        """
        x = torch.FloatTensor(x).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        mean, logvar = self.env_model.output(x, y)

        return mean, logvar

    def penalty_predict(self, rep_state_action, back_state_actions, deterministic=False):
        mean, logvar = self.predict(rep_state_action, back_state_actions)
        mean = mean.detach().cpu().numpy()
        logvar = logvar.detach().cpu().numpy()

        ensemble_model_stds = np.sqrt(np.exp(logvar))

        if deterministic:
            pred_means = mean
        else:
            pred_means = mean + np.random.normal(size=mean.shape) * ensemble_model_stds * 0.1

        num_models, batch_size, _ = pred_means.shape  # batch_size = N
        model_idxes = np.random.choice(self.env_model.elite_model_idxes, size=batch_size)
        model_idxes = np.array(model_idxes, dtype=np.int)
        batch_idxes = np.arange(0, batch_size)
        # randomly choose the predictions using the elite model
        pred_samples = pred_means[model_idxes, batch_idxes]

        pred_next_obs = pred_samples[:, :self.obs_dim]
        pred_reward = pred_samples[:, -1]
        # penalty rewards
        penalty_coeff = self.reward_penalty_coeff

        penalties = []

        for i in range(self.ensemble_size):
            for j in range(i + 1, self.ensemble_size):
                penalties.append(deepcopy(np.mean(np.square(pred_means[i, :] - pred_means[j, :]), axis=1)))
        penalties = np.array(penalties)
        penalty = self.beta * np.max(penalties, axis=0)

        penalty += np.amin(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)
        penalty_pred_reward = pred_reward - penalty_coeff * penalty

        return pred_next_obs[0], penalty_pred_reward[0], termination_fn2(pred_next_obs[:, 0], pred_next_obs[:, 1]), penalty

    def update(self, train_data):
        x = torch.Tensor(train_data["x"]).to(self.device)
        y = torch.Tensor(train_data["y"]).to(self.device)
        ground_truth = torch.Tensor(train_data["ground_truth"]).to(self.device)

        # predict with model
        predictions = self.predict(x, y)

        # training
        train_mse_losses, train_var_losses = self.model_loss(predictions, ground_truth, type_="train")

        pure_mse_loss, _, pure_mape_loss, _, pure_mae_loss, _ = self.model_loss(predictions, ground_truth, type_="eval")
        pure_mse_loss = torch.mean(pure_mse_loss)
        pure_mape_loss = torch.mean(pure_mape_loss)
        pure_mae_loss = torch.mean(pure_mae_loss)
        train_mse_loss = torch.mean(train_mse_losses)
        train_var_loss = torch.mean(train_var_losses)
        train_transition_loss = train_mse_loss + train_var_loss
        train_transition_loss += 0.01 * torch.sum(self.env_model.max_logvar) - 0.01 * torch.sum(
            self.env_model.min_logvar)
        # change
        if self.use_weight_decay:
            decay_loss = self.env_model.get_decay_loss()
            train_transition_loss += decay_loss
        # update transition model
        self.model_optimizer.zero_grad()
        train_transition_loss.backward()
        self.model_optimizer.step()
        self.sch.step()
        # compute test loss for elite model
        return {
            "loss/train_model_loss_mse": train_mse_loss.item(),
            "loss/train_model_loss_var": train_var_loss.item(),
            "loss/train_model_loss": train_transition_loss.item(),
            "loss/train_model_loss_pure_mse": pure_mse_loss.item(),
            "loss/train_model_loss_mape": pure_mape_loss.item(),
            "loss/train_model_loss_mae": pure_mae_loss.item(),
            # "loss/decay_loss": decay_loss.item() if decay_loss is not None else 0,
            "misc/max_std": self.env_model.max_logvar.mean().item(),
            "misc/min_std": self.env_model.min_logvar.mean().item()
        }

    def update_best_snapshots(self, val_losses, dim_detailed_loss_lst=None, save_model=False):
        updated = False
        for i in range(len(val_losses)):
            current_loss = val_losses[i]
            best_loss = self.best_snapshot_losses[i]
            improvement = (best_loss - current_loss) / best_loss
            if improvement > 0.01:
                self.best_snapshot_losses[i] = current_loss
                self.save_model_snapshot(i)
                updated = True

                # save the best loss of each dim
                if dim_detailed_loss_lst is not None:
                    self.best_snapshot_losses_dim_lst[i] = deepcopy(dim_detailed_loss_lst[i].cpu())
                # save the current network
                if save_model:
                    self.save_specific_model("env_model", i)
                # improvement = (best_loss - current_loss) / best_loss
                # print('epoch {} | updated {} | improvement: {:.4f} | best: {:.4f} | current: {:.4f}'.format(epoch, i, improvement, best, current))

        return updated

    def reset_best_snapshots(self):
        self.model_best_snapshots = [deepcopy(self.env_model.ensemble_models[idx].state_dict()) for idx in
                                     range(self.env_model.ensemble_size)]
        self.best_snapshot_losses = [1e10 for _ in range(self.env_model.ensemble_size)]
        self.best_snapshot_losses_dim_lst = np.ones((self.env_model.ensemble_size, self.env_model.output_dim)) * 1e10

    def save_model_snapshot(self, idx):
        self.model_best_snapshots[idx] = deepcopy(self.env_model.ensemble_models[idx].state_dict())

    def save_specific_model(self, info, model_index):
        save_dir = os.path.join(self.logger.log_path, 'model')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_dir = os.path.join(save_dir, "{}".format(info))
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        # save model with model_index
        save_path = os.path.join(model_save_dir, "env_model_" + str(model_index) + ".pt")
        torch.save(self.env_model.ensemble_models[model_index].state_dict(), save_path)
        self.logger.info_print("Transition_model_" + str(model_index) + "saved to:" + save_path)

    def load_elite_model(self, path):
        self.env_model.load_elite_model(torch.load(path, map_location=self.device))
