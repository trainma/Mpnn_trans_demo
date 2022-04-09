import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from dataset.Airquality_dataset import AirDatasetLoader
from dataset.temporal_split import temporal_signal_split_valid


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=32, periods=periods,
                            batch_size=batch_size)  # node_features=2, periods=12
        # Equals single-shot prediction

        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index, edge_attr):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        # x.shape = (batch_size,node_num,feature_size,time_steps)
        h = self.tgnn(x, edge_index, edge_attr)  # x [b, 207, 2, 12]  returns h [b, 207, 12]
        h = F.relu(h)
        h = self.linear(h)
        return h
