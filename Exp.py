import argparse
import datetime
import os
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.Multi_graph_trans import Multi_graph_trans
from model.A3TGCN2 import TemporalGNN
import torch.optim as optim
from utils.tools import EarlyStopping
from utils.tools import adjust_learning_rate

def train(epoch, adj_edge_index, adj_edge_attr, features, y, optimizer: optim, model, loss_fn):
    optimizer.zero_grad()
    output = model(features, adj_edge_index, adj_edge_attr)
    loss_train = loss_fn(output.squeeze(2), y.squeeze(2))
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


def train2(epoch, adj_edge_index, adj_edge_attr, poi_edge_index, poi_edge_attr, ST_edge_index, ST_edge_attr, features,
           y, optimizer: optim, model, loss_fn):
    optimizer.zero_grad()
    output = model(features, adj_edge_index, adj_edge_attr, poi_edge_index, poi_edge_attr, ST_edge_index, ST_edge_attr)

    loss_train = loss_fn(output.squeeze(-1), y.transpose(1, 2))
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


class Exp():
    def __init__(self, model, args, device):
        self.model = None
        self.args = args
        self.device = device
        if model == "MPNN_trans":
            self.model = Multi_graph_trans(in_channels=self.args.node_features, hidden_size=self.args.hidden,
                                           out_channels=32,
                                           d_model=256,
                                           feature_dim=9, label_len=24,
                                           num_nodes=12, window=self.args.window, dropout=self.args.dropout, num_layers=1,
                                           dec_seq_len=1,
                                           batch_size=self.args.batch_size).to(self.device)
        elif model == "A3TGCN":
            self.model = TemporalGNN(node_features=9, periods=8, batch_size=self.args.batch_size).to(self.device)

    def _build_model(self):

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        early_stopping = EarlyStopping(patience=7, verbose=True)
        path = os.path.join(self.args.checkpoints, 'run1')
        loss_fn = torch.nn.MSELoss()
        if not os.path.exists(path):
            os.makedirs(path)


    def _train(self,train_loader):

        for epoch in tqdm(range(self.args.epochs), "training:"):
            start = time.time()
            self.model.train()
            train_loss = []
            # Train for one epoch y[256 12 1]
            for batch, (X, y) in enumerate(train_loader):
                output, loss = train2(epoch, geograph_edge_index, geograph_edge_attr, poi_graph_edge_index,
                                      poi_graph_edge_attr, ST_graph_edge_index, ST_graph_edge_attr, X, y)

                train_loss.append(loss.data.item())

            print("Epoch {}: train MSE loss {:.4f}".format(epoch + 1, sum(train_loss) / len(train_loss)))
            logger.info(
                args.model + " Epoch {}: train MSE loss {:.4f}".format(epoch + 1, sum(train_loss) / len(train_loss)))
            writer.add_scalar('train_loss', sum(train_loss) / len(train_loss), epoch)
            # Evaluate on validation set
            model.eval()
            total_loss = []
            mae_l, mse_l, rmse_l, mape_l, mspe_l = [], [], [], [], []
    def _valid(self):

    def _test(self):
