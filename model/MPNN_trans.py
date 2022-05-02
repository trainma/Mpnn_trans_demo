import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from model.informer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from model.informer.decoder import Decoder, DecoderLayer
from model.informer.attn import FullAttention, ProbAttention, AttentionLayer
from model.informer.embed import DataEmbedding


##

class MPNN_trans(nn.Module):
    def __init__(self, nfeat, nhid, nout, nodes, window, dropout, batch_size, label_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 attn='prob', embed='fixed', freq='h', activation='gelu', output_attention=False,
                 distil=True, mix=True, device=torch.device('cuda')):
        super(MPNN_trans, self).__init__()
        self.label_len = label_len
        self.window = window
        self.nodes = nodes
        self.nhid = nhid
        self.nfeat = nfeat
        self.nout = nout
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.batch_size = batch_size

        self.bn1 = nn.BatchNorm2d(nhid)
        self.bn2 = nn.BatchNorm2d(nhid)
        self.device = torch.device('cuda')
        self.pred_len = nout
        self.attn = attn
        self.output_attention = output_attention
        self.enc_in = nhid * 2
        self.dec_in = nhid * 2
        self.enc_embedding = DataEmbedding(self.enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, d_model, embed, freq, dropout)
        self.label_len = label_len
        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, nout, bias=True)

    def forward(self, X, adj_edge_index, adj_edge_attr, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        lst = list()
        # X [bz nodes feature time]
        X = X.permute(0, 3, 1, 2)  # [bz time node feature]
        skip = X
        skip = torch.transpose(skip, 1, 2).reshape(-1, self.window,
                                                   self.nfeat)
        # [bz wd,n_nodes,feats]
        x = self.relu(self.conv1(X, adj_edge_index, adj_edge_attr))  # [256 12 12 128]
        x = x.permute(0, 3, 2, 1)  # [256 128 12 12]
        # [Nbz Cfeatures H Wseqlen] [256 128 12 12]
        x = self.bn1(x)
        x = x.permute(0, 3, 2, 1)  # [256 12 12 128]
        x = self.dropout(x)
        lst.append(x)

        x = self.relu(self.conv2(x, adj_edge_index, adj_edge_attr))
        x = x.permute(0, 3, 2, 1)
        x = self.bn2(x)
        x = x.permute(0, 3, 2, 1)
        x = self.dropout(x)
        lst.append(x)
        x = torch.cat(lst, dim=-1)
        # x[bz wd node feature]
        x = torch.transpose(x, 0, 1)  # [wd bz node feature]
        x = x.permute(2, 3, 0, 1).contiguous().view(-1, self.window, self.nhid*2)
        # [bz seq_len feature] [bz wd feature] [768 64 128]
        x_enc = x
        x_dec = torch.zeros(x.shape[0], self.pred_len, self.nhid*2).to(self.device)
        dec_inp = torch.cat((x_enc[:, -self.label_len:, :], x_dec), dim=1)
        enc_out = self.enc_embedding(
            x_enc)  # x_enc.shape([128(bz), 96(seq_len), 6(feature_dim)]) x_mark_enc.shape([64, 96, 4]) enc_out [64 96 512]
        # x_enc [bz seq_len feature]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)  # [64 48 512] attns[None None]
        # enc_out [128 48 512]
        dec_out = self.dec_embedding(dec_inp)  # x_dec [64 72 7] x_mark_dec[64 72 4] dec_out[64 72 512]
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)  # [64 72 512]
        dec_out = dec_out.view(self.batch_size, self.nodes, self.label_len + 1, -1)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, :, -self.pred_len:, :], attns
        else:
            return dec_out[:, :, -self.pred_len:, :]
