import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import scipy.sparse as sp

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import os
from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARIMA
from copy import deepcopy


class MPNNLSTM(nn.Module):
    def __init__(self, nfeat, nhid, nout, nodes, window, dropout,batch_size):
        super(MPNNLSTM, self).__init__()
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

        self.rnn1 = nn.LSTM(2 * nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)

        # self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc1 = nn.Linear(2 * nhid + window * nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, X, adj_edge_index, adj_edge_attr):
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

        x = torch.transpose(x, 0, 1)  # [wd bz node feature]
        x = x.contiguous().view(self.window, -1, x.shape[-1])
        x, (hn1, cn1) = self.rnn1(x)  # [7(seq_len-time windows) 5*129(batch_size) 128(d_models)]

        out2, (hn2, cn2) = self.rnn2(x)

        # print(self.rnn2._all_weights)
        x = torch.cat([hn1[0, :, :], hn2[0, :, :]], dim=1)
        skip = skip.reshape(skip.shape[0], -1)
        x = torch.cat([x, skip], dim=1)
        # --------------------------------------
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = x.view(self.batch_size, -1, self.nout)


        return x
