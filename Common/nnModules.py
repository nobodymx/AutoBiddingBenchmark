import torch
import torch.nn as nn
from typing import Union
from Common.act_fn import get_act_fn
import math
import numpy as np


class MLP(nn.Module):
    def __init__(
            self, input_dim: int,
            output_dim: int,
            hidden_dims: Union[int, list],
            act_fn="swish",
            out_act_fn="identity",
            device="cpu",
    ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        hidden_dims = [input_dim] + hidden_dims
        self.networks = []
        act_fn = get_act_fn(act_fn)
        out_act_fn = get_act_fn(out_act_fn)

        for i in range(len(hidden_dims) - 1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i + 1]
            curr_network = torch.nn.Linear(curr_shape, next_shape)
            self.networks.extend([curr_network, act_fn()])
        final_network = torch.nn.Linear(hidden_dims[-1], output_dim)
        self.networks.extend([final_network, out_act_fn()])
        self.networks = nn.Sequential(*self.networks)
        self.to(device)
        self._weights = [net.weight for net in self.networks if isinstance(net, torch.nn.modules.linear.Linear)]

    def forward(self, input):
        return self.networks(input)

    @property
    def weights(self):
        return self._weights

    def to(self, device):
        self.networks = self.networks.to(device)


class FusionEmbeddingNet(nn.Module):
    def __init__(self,
                 dense_input_dim=11,
                 id_onehot_input_dim=182,
                 id_onehot_output_dim=8,
                 id_onehot_hidden_dim=None,
                 common_hidden_dim=None,
                 output_dim=50,
                 act_fn="relu",
                 out_act_fn="identity",
                 device="cpu",
                 id_onehot_output_act_fn="tanh",
                 ):
        super(FusionEmbeddingNet, self).__init__()
        self.dense_dim = dense_input_dim
        self.id_onehot_dim = id_onehot_input_dim
        self.id_onehot_network = MLP(id_onehot_input_dim, id_onehot_output_dim, id_onehot_hidden_dim,
                                     act_fn=act_fn, out_act_fn=id_onehot_output_act_fn, device=device)
        self.common_network = MLP(dense_input_dim + id_onehot_output_dim, output_dim, common_hidden_dim,
                                  act_fn=act_fn, out_act_fn=out_act_fn, device=device)

        self.to(device)
        self._weights = []
        self._weights.extend(self.id_onehot_network.weights)
        self._weights.extend(self.common_network.weights)

    def forward(self, obs):
        dense_x, id_x = obs[..., :self.dense_dim], obs[..., self.dense_dim:]
        embedding = self.id_onehot_network(id_x)
        merge = torch.cat([dense_x, embedding], -1)
        y = self.common_network(merge)
        return y

    @property
    def weights(self):
        return self._weights

    def to(self, device):
        self.id_onehot_network.to(device)
        self.common_network.to(device)


class ActorProb(nn.Module):
    def __init__(self, backbone, dist_net, device="cpu"):
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone
        self.dist_net = dist_net
        self.to()

    def get_dist(self, obs):
        # obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist

    def to(self):
        self.backbone.to(self.device)
        self.dist_net.to(self.device)

    def forward(self, obs, deterministic: bool):
        logits = self.backbone(obs)
        return self.dist_net(logits, deterministic)


class DDPGActorCritic(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 id_cate_dim: int,
                 embedding_dim: int,
                 base_hidden_dims: Union[int, list],
                 embedding_hidden_dims: Union[int, list],
                 act_fn="swish",
                 embedding_out_act_fn="tanh",
                 device="cpu",
                 ):
        super(DDPGActorCritic, self).__init__()
        self.embedding_mlp = MLP(id_cate_dim, embedding_dim, embedding_hidden_dims, act_fn=act_fn,
                                 out_act_fn=embedding_out_act_fn, device=device)
        self.fusion = MLP(embedding_dim + input_dim, output_dim, base_hidden_dims,
                          act_fn=act_fn, device=device)

    def forward(self, id_cate, x):
        embedding = self.embedding_mlp(id_cate)
        y = self.fusion(torch.cat([embedding, x]))
        return y


class DeterministicActor(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 id_cate_dim: int,
                 embedding_dim: int,
                 base_hidden_dims: Union[int, list],
                 embedding_hidden_dims: Union[int, list],
                 act_fn="swish",
                 out_act_fn="identity",
                 embedding_out_act_fn="tanh",
                 device="cpu",
                 ):
        super(DeterministicActor, self).__init__()
        self.imme_dim = base_hidden_dims[-1]
        self.embedding_mlp = MLP(id_cate_dim, embedding_dim, embedding_hidden_dims, act_fn=act_fn,
                                 out_act_fn=embedding_out_act_fn, device=device)
        self.base_mlp = MLP(input_dim + embedding_dim, self.imme_dim, base_hidden_dims, act_fn=act_fn,
                            out_act_fn=out_act_fn,
                            device=device)
        self.out_net = nn.Linear(self.imme_dim, output_dim)
        self.device = device
        self.to()

    def forward(self, id_cate, x):
        embedding = self.embedding_mlp(id_cate)
        x = self.base_mlp(torch.cat([x, embedding], dim=-1))
        x = self.out_net(x)
        return x

    def to(self):
        self.out_net = self.out_net.to(self.device)


class DeterministicCritic(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 base_hidden_dims: Union[int, list],
                 act_fn="swish",
                 out_act_fn="identity",
                 device="cpu",
                 ):
        super(DeterministicCritic, self).__init__()
        self.base_mlp = MLP(input_dim, base_hidden_dims[-1], base_hidden_dims, act_fn=act_fn,
                            out_act_fn=out_act_fn,
                            device=device)
        self.out_net = nn.Linear(base_hidden_dims[-1], output_dim)
        self.device = device
        self.to()

    def forward(self, states, actions):
        x = self.base_mlp(torch.cat([states, actions], 1))
        x = self.out_net(x)
        return x

    def to(self):
        self.out_net = self.out_net.to(self.device)


class GaussianMLP(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 id_cate_dim: int,
                 embedding_dim: int,
                 base_hidden_dims: Union[int, list],
                 mu_sigma_hidden_dims: Union[int, list],
                 embedding_hidden_dims: Union[int, list],
                 act_fn="swish",
                 out_act_fn="identity",
                 embedding_out_act_fn="tanh",
                 device="cpu",
                 sigma_min=-20,
                 sigma_max=0
                 ):
        super(GaussianMLP, self).__init__()
        self.imme_dim = base_hidden_dims[-1]
        self.embedding_mlp = MLP(id_cate_dim, embedding_dim, embedding_hidden_dims, act_fn=act_fn,
                                 out_act_fn=embedding_out_act_fn, device=device)
        self.base_mlp = MLP(input_dim + embedding_dim, self.imme_dim, base_hidden_dims, act_fn=act_fn,
                            out_act_fn=out_act_fn,
                            device=device)
        self.mu_mlp = nn.Linear(self.imme_dim, output_dim)
        self.sigma_mlp = nn.Linear(self.imme_dim, output_dim)
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        # self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self.device = device

        self.to()

    def forward(self, id_cate, x):
        embedding = self.embedding_mlp(id_cate)
        x = self.base_mlp(torch.cat([x, embedding], dim=-1))
        mu = self.mu_mlp(x)
        sigma = torch.clamp(self.sigma_mlp(x), min=self._sigma_min, max=self._sigma_max).exp()
        return mu, sigma

    def to(self):
        self.mu_mlp = self.mu_mlp.to(self.device)
        self.sigma_mlp = self.sigma_mlp.to(self.device)
        # self.sigma_param = self.sigma_param.to(self.device)


class DiagGaussian(nn.Module):
    def __init__(
            self,
            latent_dim,
            output_dim,
            unbounded=False,
            conditioned_sigma=False,
            max_mu=1.0,
            sigma_min=-20,
            sigma_max=2
    ):
        super().__init__()
        self.mu = nn.Linear(latent_dim, output_dim)
        self._c_sigma = conditioned_sigma
        self.sigma = nn.Linear(latent_dim, output_dim)
        self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self._unbounded = unbounded
        self._max = max_mu
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def forward(self, logits, deterministic: bool):
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=self._sigma_min, max=self._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        if deterministic:
            action = mu
        else:
            action = mu + sigma * torch.normal(torch.zeros(sigma.shape, dtype=sigma.dtype, device=sigma.device),
                                               torch.ones(sigma.shape, dtype=sigma.dtype, device=sigma.device))
        var = (sigma ** 2)
        log_scale = sigma.log()
        log_prob = -((action - mu) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class SelfAttention(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, kq_dim: int,
                 v_dim: int, v_hidden_layers=None, act_fn="relu", out_act_fn="identity", device="cpu"):
        super(SelfAttention, self).__init__()
        # embedding network
        if v_hidden_layers is None:
            v_hidden_layers = [100, 50, 50]
        self.embedding_network = torch.nn.Linear(input_dim, embedding_dim, bias=False)
        # key network
        self.K = torch.nn.Linear(embedding_dim, kq_dim, bias=False)
        # query network
        self.Q = torch.nn.Linear(embedding_dim, kq_dim, bias=False)
        # value network
        self.V = torch.nn.Linear(embedding_dim, v_dim, bias=False)

        self.to(device)
        self._weights = []
        self._weights.extend(self.embedding_network.weight)
        self._weights.extend(self.K.weight)
        self._weights.extend(self.Q.weight)
        self._weights.extend(self.V.weight)

    def forward(self, x):
        """
        :param x: [batch, channel, feature_dim]
        :return:
        """
        # embeddings
        x = self.embedding_network(x)  # batch * channel * emb_dim
        # keys
        keys = self.K(x)  # batch * channel * kq_dim
        # queries
        queries = self.Q(x)  # batch * channel * kq_dim
        # values
        values = self.V(x)  # batch * channel * v_dim
        # output
        alpha = torch.matmul(queries, keys.permute(0, 2, 1))  # batch * channel * channel
        alpha = torch.softmax(alpha, 2)  # batch * channel * channel
        y = torch.matmul(alpha, values)  # batch * channel * v_dim

        return y

    def to(self, device):
        self.embedding_network = self.embedding_network.to(device)
        self.K = self.K.to(device)
        self.Q = self.Q.to(device)

    def weights(self):
        return self._weights


class ScaledDotProductAttention(nn.Module):
    def __init__(self, kq_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.kq_dim = kq_dim

    def forward(self, Q, K, V):  # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.kq_dim)  # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiheadSelfAttention(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, kq_dim: int,
                 v_dim: int, v_hidden_layers=None, act_fn="relu", out_act_fn="identity",
                 head_extraction_layers=None,
                 device="cpu", head_num=1):
        super(MultiheadSelfAttention, self).__init__()
        if head_extraction_layers is None:
            head_extraction_layers = [100, 100, 50, 10]
        if v_hidden_layers is None:
            v_hidden_layers = [100, 50, 50]
        self.head_num = head_num
        self.device = device
        self.self_attention_networks = [SelfAttention(input_dim, embedding_dim, kq_dim,
                                                      v_dim, v_hidden_layers=v_hidden_layers, act_fn=act_fn,
                                                      out_act_fn=out_act_fn, device=device)
                                        for _ in range(self.head_num)]

        # feature extraction across heads
        self.feature_extraction_across_heads = torch.nn.Linear(head_num, 1, bias=False)
        self.to()
        self._weights = [net.weights for net in self.self_attention_networks]
        self._weights.extend(self.feature_extraction_across_heads.weight)
        for i in range(self.head_num):
            self.add_module("attention_head_{}".format(i), self.self_attention_networks[i])

    def forward(self, x):
        """
        :param x: [batch, channel, feature_dim]
        :return:
        """
        y = [self.self_attention_networks[i](x).unsqueeze(3) for i in
             range(self.head_num)]  # head_num (batch, channel, feature_dim)
        y = torch.cat(y, 3)  # batch * channel * feature_dim * head num
        y = self.feature_extraction_across_heads(y * 10)  # batch * channel * feature_dim * 1
        y = y.squeeze(3)  # batch * channel * feature_dim

        return y

    def to(self):
        self.feature_extraction_across_heads = self.feature_extraction_across_heads.to(self.device)

    def weights(self):
        return self._weights


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, kq_dim: int, num_heads: int,
                 v_dim: int, device="cpu"):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(input_dim, kq_dim * num_heads, )
        self.W_K = nn.Linear(input_dim, kq_dim * num_heads, )
        self.W_V = nn.Linear(input_dim, v_dim * num_heads, )
        self.fc = nn.Linear(num_heads * v_dim, input_dim, )
        self.input_dim = input_dim
        self.kq_dim = kq_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.device = device

        self.to()

    def to(self):
        self.W_Q = self.W_Q.to(self.device)
        self.W_K = self.W_K.to(self.device)
        self.W_V = self.W_V.to(self.device)

    def forward(self, x):
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = x, x.size(0)
        Q = self.W_Q(x).view(batch_size, -1, self.num_heads, self.kq_dim).transpose(1,
                                                                                    2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(x).view(batch_size, -1, self.num_heads, self.kq_dim).transpose(1,
                                                                                    2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(x).view(batch_size, -1, self.num_heads, self.v_dim).transpose(1,
                                                                                   2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        context, attn = ScaledDotProductAttention(self.kq_dim)(Q, K,
                                                               V, )  # context: [batch_size, n_heads, len_q, d_v] # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.num_heads * self.v_dim)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.input_dim).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, device="cpu"):
        super(PoswiseFeedForwardNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, ),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim, ),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim, ))
        self.device = device

        self.to()

    def to(self):
        self.fc = self.fc.to(self.device)

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.input_dim).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, input_dim: int, kq_dim: int, num_heads: int,
                 v_dim: int, hidden_dim: int, device="cpu"):
        super(EncoderLayer, self).__init__()
        self.input_dim = input_dim
        self.kq_dim = kq_dim
        self.num_heads = num_heads
        self.v_dim = v_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.enc_self_attn = MultiHeadAttention(input_dim, kq_dim, num_heads, v_dim, device=device)  # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet(input_dim, hidden_dim, device=device)  # 前馈神经网络

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, src_len, d_model]
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V            # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs)  # enc_outputs: [batch_size, src_len, d_model],
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, num_encoder_layer: int, input_dim: int, kq_dim: int, num_heads: int,
                 v_dim: int, hidden_dim: int, device="cpu"):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(input_dim, kq_dim, num_heads, v_dim, hidden_dim, device=device) for _ in
             range(num_encoder_layer)])

    def forward(self, x):  # enc_inputs: [batch_size, src_len, d_model]
        enc_self_attns = []
        for layer in self.layers:
            x, enc_self_attn = layer(x, )  # enc_outputs :   [batch_size, src_len, d_model],
            # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        return x, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, input_dim: int, kq_dim: int, num_heads: int,
                 v_dim: int, hidden_dim: int, device="cpu"):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(input_dim, kq_dim, num_heads, v_dim, device=device)
        self.dec_enc_attn = MultiHeadAttention(input_dim, kq_dim, num_heads, v_dim, device=device)
        self.pos_ffn = PoswiseFeedForwardNet(input_dim, hidden_dim, device=device)

    def forward(self, dec_inputs, enc_outputs):  # dec_inputs: [batch_size, tgt_len, d_model]
        # enc_outputs: [batch_size, src_len, d_model]
        # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, )  # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs, )  # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)  # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, num_decoder_layers: int,
                 input_dim: int, kq_dim: int, num_heads: int,
                 v_dim: int, hidden_dim: int, device="cpu"
                 ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(input_dim, kq_dim, num_heads, v_dim, hidden_dim,
                                                  device=device) for _ in range(num_decoder_layers)])

    def forward(self, dec_outputs, enc_outputs):  # dec_inputs: [batch_size, tgt_len]
        # enc_intpus: [batch_size, src_len]
        # enc_outputs: [batsh_size, src_len, d_model]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:  # dec_outputs: [batch_size, tgt_len, d_model]
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
