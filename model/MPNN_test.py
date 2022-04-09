from . import mpnn_lstm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from model.Ttransformer.positional_encoding import *


class AR(nn.Module):

    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        # x: [batch, window, n_multiv]
        x = self.linear(x)  # x: [batch, n_multiv, 1]
        return x


class test_model(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, out_channels: int
                 , num_nodes: int, window: int, num_layers, dropout: float, dec_seq_len, batch_size):
        super(test_model, self).__init__()
        self.window = window
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self._convolution_1 = GCNConv(self.in_channels, self.hidden_size)
        self._convolution_2 = GCNConv(self.hidden_size, self.hidden_size)

        self.src_mask = None
        feature_size = 256
        # d_model 词嵌入维度

        self.d_model = feature_size
        self.dropout = dropout
        self.max_len = 100

        self.local = context_embedding(self.d_model, self.d_model, 1)

        # ----------whether use learnable position encoding---------#
        self.pos_encoder = PositionalEncoding(self.d_model)

        # ----------whether transformer encoder layer with batch_norm---------#
        self.input_project = nn.Linear(self.hidden_size, self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=16, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=16, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=dec_seq_len)

        self.tmp_out = nn.Linear(feature_size, 1)
        self.src_key_padding_mask = None
        self.transformer = nn.Transformer(d_model=self.d_model,
                                          nhead=8,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=dec_seq_len,
                                          dropout=dropout, )

        self.dropout = nn.Dropout(dropout)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, X, edge_index, edge_attr):
        X = X.permute(0, 1, 3, 2)  # X:[bz,node,timesteps,node_features]
        X = self._convolution_1(X, edge_index, edge_attr)
        X = self.dropout(X)
        X = self._convolution_2(X, edge_index, edge_attr)
        X = self.dropout(X)  # [16 12 12 32] [bz,node,timesteps,node_features]

        time_steps, feature_dim = X.shape[2], X.shape[3]
        src = X.view(-1, time_steps, feature_dim)  # src[192bz 12ts 32fd]
        src = src.permute(1, 0, 2)  # [time_steps, batch_size,feature_dim] [12 192 32]
        tgt = src[-3:, :, :]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 初始化掩码张量
        mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(device)
        # src [12 192 32]
        src = self.input_project(src) * math.sqrt(self.d_model)  # [12ts 192bz 256fd]
        tgt = self.input_project(tgt) * math.sqrt(self.d_model)
        src = self.local(src.permute(1, 2, 0))  # [192 256 12]
        src = src.permute(2, 0, 1)  # [12(seq_len) 192(bz) 256]
        # torch.Size([168, 16, 256])->torch.Size([168, 16, pos_encoder.featuresize(256)])
        src = self.pos_encoder(src)  # torch.Size([168,16,64 ])
        tgt = self.pos_encoder(tgt)  # torch,Size(3 16 64)

        x = self.transformer(src=src,
                             tgt=tgt,
                             tgt_mask=mask)
        x = x.view(-1, self.batch_size, 256)
        transformer_out = self.tmp_out(x)[0, :, :]
        return transformer_out
