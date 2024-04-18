import torch
import torch.nn as nn
from Models.transformer import TransformerModel
from Common.nnModules import MLP


class PE_Model(nn.Module):
    def __init__(self, input_dim=20, output_dim=8, hidden_layer_num=5, hidden_io=100,
                 trans_output_dim=100,
                 embedding_dim=100, head_num=8, kq_dim=100, v_hidden_layers=None,
                 head_extraction_layers=None, act_fn="swish", out_act_fn="identity",
                 mlp_hidden_layers=None,
                 trans_mlp_hidden_layers=None,
                 total_mlp_hidden_layers=None,
                 device="cpu"):
        super(PE_Model, self).__init__()
        self.device = device
        self.transformer_model = TransformerModel(input_dim=input_dim,
                                                  output_dim=trans_output_dim,
                                                  hidden_layer_num=2,
                                                  hidden_io=hidden_io,
                                                  embedding_dim=embedding_dim,
                                                  head_num=head_num,
                                                  kq_dim=kq_dim,
                                                  v_hidden_layers=v_hidden_layers,
                                                  head_extraction_layers=head_extraction_layers,
                                                  act_fn=act_fn,
                                                  out_act_fn=out_act_fn,
                                                  device=device,
                                                  )
        self.self_process_model = torch.nn.Linear(input_dim, trans_output_dim, bias=False)
        self.PI_process_model = torch.nn.Linear(trans_output_dim * 2, trans_output_dim * 2, bias=False)
        self.total_process_model = MLP(trans_output_dim * 3, output_dim, total_mlp_hidden_layers, act_fn=act_fn,
                                       out_act_fn=out_act_fn, device=device)

        self.to()

    def forward(self, x, y):
        # x: [batch, input_dim]
        # y: [batch, N-1, input_dim]
        x = self.self_process_model(x)  # x: [batch, output_dim]
        y = self.transformer_model(y)  # y: [batch, N-1, trans_output_dim]
        # y *= 10
        # y_floor = y[:, 0].type(torch.int)
        # y_floor = torch.stack([y_floor for i in range(y.shape[1])], dim=1)
        # y -= y_floor
        y_1 = torch.mean(y, 1)  # y_1: [batch, trans_output_dim]
        y_2 = torch.max(y, 1).values  # y_2: [batch, trans_output_dim]
        y = torch.cat([y_1, y_2], 1)
        y = self.PI_process_model(y)  # y: [batch, trans_output_dim]
        z = torch.cat([x, y], dim=-1)  # z: [batch, output_dim + trans_output_dim]
        z = self.total_process_model(z)  # z: [batch, output_dim]
        return z

    def to(self):
        self.self_process_model = self.self_process_model.to(self.device)
        self.PI_process_model = self.PI_process_model.to(self.device)