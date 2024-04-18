from Common.nnModules import MultiheadSelfAttention, Encoder, Decoder
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim=20, output_dim=8, hidden_layer_num=5, hidden_io=100,
                 embedding_dim=100, head_num=8, kq_dim=100, v_hidden_layers=None,
                 head_extraction_layers=None, act_fn="swish", out_act_fn="identity", device="cpu", ):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.kq_dim = kq_dim
        self.v_hidden_layers = v_hidden_layers
        self.head_extraction_layers = head_extraction_layers
        self.output_dim = output_dim
        self.hidden_io = hidden_io
        self.hidden_layer_num = hidden_layer_num
        self.head_num = head_num
        self.act_fn = act_fn
        self.out_act_fn = out_act_fn
        self.device = device

        # the dimension of the network
        self.input_seq_dim = [self.input_dim]
        self.output_seq_dim = []
        for i in range(self.hidden_layer_num - 1):
            self.input_seq_dim.append(self.hidden_io)
            self.output_seq_dim.append(self.hidden_io)
        self.output_seq_dim.append(self.output_dim)

        # the network
        self.networks = [MultiheadSelfAttention(self.input_seq_dim[i], self.embedding_dim, self.kq_dim,
                                                self.output_seq_dim[i], v_hidden_layers=self.v_hidden_layers,
                                                act_fn=self.act_fn, out_act_fn=self.out_act_fn,
                                                head_extraction_layers=self.head_extraction_layers,
                                                device=self.device, head_num=self.head_num
                                                )
                         for i in range(self.hidden_layer_num)]
        for i in range(self.hidden_layer_num):
            self.add_module("attention_layer_{}".format(i), self.networks[i])
        self._weight = [net.weights for net in self.networks]

    def forward(self, x):
        """
        :param x: [batch, channel, input_dim]
        :return: [batch, channel, output_dim]
        """
        for i in range(self.hidden_layer_num):
            x = self.networks[i](x)
        return x

    def weights(self):
        return self._weight


class Transformer(nn.Module):
    def __init__(self, num_decoder_layers: int, num_encoder_layer: int,
                 input_dim: int, kq_dim: int, num_heads: int,
                 v_dim: int, hidden_dim: int, device="cpu"):
        super(Transformer, self).__init__()
        self.Encoder = Encoder(num_encoder_layer, input_dim, kq_dim, num_heads,
                               v_dim, hidden_dim, device=device)
        self.Decoder = Decoder(num_decoder_layers,
                               input_dim, kq_dim, num_heads,
                               v_dim, hidden_dim, device=device)

    def forward(self, enc_inputs, dec_inputs):  # enc_inputs: [batch_size, src_len]
        # dec_inputs: [batch_size, tgt_len]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)  # enc_outputs: [batch_size, src_len, d_model],
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, enc_inputs, enc_outputs)  # dec_outpus    : [batch_size, tgt_len, d_model],
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        return dec_outputs
