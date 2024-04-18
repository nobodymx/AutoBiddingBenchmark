import torch
from Models.PE_model import PE_Model
import torch.nn as nn
import torch.nn.functional as F


class PEEnsembleModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_num, hidden_io, trans_output_dim, embedding_dim,
                 head_num, v_hidden_layers, head_extraction_layers, kq_dim, mlp_hidden_layer,
                 ensemble_size=4,
                 num_elite=5, decay_weights=None,
                 trans_mlp_hidden_layers=None,
                 total_mlp_hidden_layers=None,
                 act_fn="swish", out_act_fn="identity",
                 device="cpu"):
        super(PEEnsembleModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mean_dim = int(output_dim / 2)
        self.var_dim = int(output_dim / 2)
        self.device = device

        self.act_fn = act_fn
        self.out_act_fn = out_act_fn
        self.ensemble_size = ensemble_size
        self.decay_weights = decay_weights
        self.num_elite = num_elite
        self.device = device

        self.ensemble_models = [PE_Model(input_dim=input_dim,
                                         output_dim=output_dim,
                                         hidden_layer_num=hidden_layer_num,
                                         hidden_io=hidden_io,
                                         trans_output_dim=trans_output_dim,
                                         embedding_dim=embedding_dim,
                                         head_num=head_num,
                                         kq_dim=kq_dim,
                                         v_hidden_layers=v_hidden_layers,
                                         head_extraction_layers=head_extraction_layers,
                                         act_fn=act_fn,
                                         out_act_fn=out_act_fn,
                                         mlp_hidden_layers=mlp_hidden_layer,
                                         trans_mlp_hidden_layers=trans_mlp_hidden_layers,
                                         total_mlp_hidden_layers=total_mlp_hidden_layers,
                                         device=device) for i in range(self.ensemble_size)]

        for i in range(self.ensemble_size):
            self.add_module("model_{}".format(i), self.ensemble_models[i])

        self.elite_model_idxes = torch.tensor([i for i in range(num_elite)])
        self.max_logvar = nn.Parameter((torch.ones((1, self.var_dim)).float() / 2).to(device), requires_grad=True)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.var_dim)).float() * 10).to(device), requires_grad=True)

        self.register_parameter("max_logvar", self.max_logvar)
        self.register_parameter("min_logvar", self.min_logvar)

        self.to(device)

    def output(self, x, y):
        outputs = [self.ensemble_models[i](x, y) for i in range(self.ensemble_size)]
        predictions = torch.stack(outputs)  # num_ensemble, N=batch, features
        mean = predictions[:, :, :self.mean_dim]
        logvar = predictions[:, :, self.var_dim:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def get_decay_loss(self):
        decay_losses = []
        for model_net in self.ensemble_models:
            curr_net_decay_losses = [decay_weight * torch.sum(torch.square(weight)) for decay_weight, weight in
                                     zip(self.decay_weights, model_net.weights_mean)]
            curr_net_decay_losses += [decay_weight * torch.sum(torch.square(weight)) for decay_weight, weight in
                                      zip(self.decay_weights, model_net.weights_logvar)]
            decay_losses.append(torch.sum(torch.stack(curr_net_decay_losses)))
        return torch.sum(torch.stack(decay_losses))

    def load_state_dicts(self, state_dicts):
        for i in range(self.ensemble_size):
            self.ensemble_models[i].load_state_dict(state_dicts[i])

    def load_model(self, model):
        for i in range(self.ensemble_size):
            self.ensemble_models[i].load_state_dict(model.__getattr__('model_' + str(i)).state_dict())

    def load_elite_model(self, elite_static_dict):
        """
        load the elite to model 0, fixed elite model index = 0
        :param elite_static_dict:
        :return:
        """
        self.elite_model_idxes = torch.Tensor([i for i in range(1)])
        for i in range(self.ensemble_size):
            self.ensemble_models[i].load_state_dict(elite_static_dict)
