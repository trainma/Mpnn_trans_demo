# coding: utf-8
import os
import time
import argparse
import networkx as nx
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from math import ceil
from tqdm import tqdm
import itertools
import pandas as pd
from model.mpnn_lstm import MPNNLSTM
from utils.log import Log
from model.mpnn_lstm import RecurrentGCN
# train_trans_log = Log("train", "./log/MPNN_trans.txt")
# logger = train_trans_log.get_log()
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from dataset.Airquality_dataset import AirDatasetLoader, AirDatasetLoader2
from dataset.temporal_split import temporal_signal_split_valid
from utils.metric import AverageMeter
from model.MPNN_test import test_model
from model.A3TGCN import TemporalGNN

writer = SummaryWriter('runs/scalar_example')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--model', type=str, default='MPNN_LSTM',
                    help='choose from MPNN_LSTM')
parser.add_argument('--batch-size', type=int, default=256,
                    help='Size of batch.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate.')
parser.add_argument('--num_layers', type=int, default=1,
                    help='The num of Transformer Decoder layer')
parser.add_argument('--window', type=int, default=12,
                    help='Size of window for features.')
parser.add_argument('--graph-window', type=int, default=7,
                    help='Size of window for graphs in MPNN LSTM.')
parser.add_argument('--recur', default=False,
                    help='True or False.')
parser.add_argument('--early-stop', type=int, default=1000,
                    help='How many epochs to wait before stopping.')
parser.add_argument('--start-exp', type=int, default=15,
                    help='The first day to start the predictions.')
parser.add_argument('--ahead', type=int, default=14,
                    help='The number of days ahead of the train set the predictions should reach.')
parser.add_argument('--sep', type=int, default=10,
                    help='Seperator for validation and train set.')
parser.add_argument('--shuffle', type=bool, default=False,
                    help='whether to shuffle dataset')
parser.add_argument('--ratio', type=list, default=[0.8, 0.1, 0.1],
                    help='the split ratio of train/valid/test')
parser.add_argument('--pred_len', type=int, default=12,
                    help='the pred_len of the model')
parser.add_argument('--num_nodes', type=int, default=12,
                    help='the pred_len of the model')
parser.add_argument('--node_features', type=int, default=11,
                    help='the pred_len of the model')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


def CreateDataloader(TrainDataset, ValidDataset, TestDataset, DEVICE=device,
                     batch_size=args.batch_size, shuffle=args.shuffle):
    def temp_dataloader(temp_dataset):
        temp_input = np.array(temp_dataset.features)  # (27399, 207, 2, 12)
        temp_target = np.array(temp_dataset.targets)  # (27399, 207, 12)
        temp_x_tensor = torch.from_numpy(temp_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
        temp_target_tensor = torch.from_numpy(temp_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
        temp_dataset_new = torch.utils.data.TensorDataset(temp_x_tensor, temp_target_tensor)
        temp_loader = torch.utils.data.DataLoader(temp_dataset_new, batch_size=batch_size, shuffle=shuffle,
                                                  drop_last=True)
        return temp_loader

    train_loader = temp_dataloader(TrainDataset)
    valid_loader = temp_dataloader(ValidDataset)
    test_loader = temp_dataloader(TestDataset)
    return train_loader, valid_loader, test_loader


def train(epoch, adj_edge_index, adj_edge_attr, features, y):
    optimizer.zero_grad()
    output = model(features, adj_edge_index, adj_edge_attr)
    loss_train = F.mse_loss(output, y)
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


def model_test(adj_edge_index, adj_edge_attr, features, y):
    output = model(features, adj_edge_index, adj_edge_attr)
    loss_test = F.mse_loss(output, y)
    return output, loss_test


if __name__ == "__main__":
    loader = AirDatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=args.window, num_timesteps_out=12)

    print("Dataset type:  ", dataset)
    print("Number of samples / sequences: ", len(set(dataset)))

    train_dataset, valid_dataset, test_dataset = temporal_signal_split_valid(dataset, ratio=[0.8, 0.1, 0.1])
    train_loader, valid_loader, test_loader = CreateDataloader(train_dataset, valid_dataset, test_dataset)

    for snapshot in train_dataset:
        static_edge_index = snapshot.edge_index.to(device)
        static_edge_attr = snapshot.edge_attr.to(device)
        break

    if args.model == "MPNN_LSTM":

        # model = test_model(in_channels=args.node_features, hidden_size=args.hidden, out_channels=32, num_nodes=12,
        #                    window=args.window, dropout=args.dropout, num_layers=1, dec_seq_len=1,
        #                    batch_size=args.batch_size).to(device)
        model = TemporalGNN(node_features=11, periods=args.window, batch_size=args.batch_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in tqdm(range(args.epochs), "training:"):
            start = time.time()
            model.train()
            train_loss = []
            # Train for one epoch
            for batch, (X, y) in enumerate(train_loader):
                output, loss = train(epoch, static_edge_index, static_edge_attr, X, y)
                train_loss.append(loss.data.item())
            print("Epoch {}: train MSE loss {:.4f}".format(epoch + 1, sum(train_loss) / len(train_loss)))
            writer.add_scalar('train_loss', sum(train_loss) / len(train_loss), epoch)
            # Evaluate on validation set
            model.eval()
            total_loss = []
            for batch, (X, y) in enumerate(valid_loader):
                y_hat = model(X, static_edge_index, static_edge_attr)
                val_loss = F.mse_loss(y_hat, y)
                total_loss.append(val_loss.detach().cpu().item())

            print("Epoch {}: val MSE loss {:.4f}".format(epoch + 1, sum(total_loss) / len(train_loss)))
            writer.add_scalar('val_loss', sum(total_loss) / len(total_loss), epoch)

        # # Print results
        # if (epoch % 50 == 0):
        #     # print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg), "time=", "{:.5f}".format(time.time() - start))
        #     # logger.info("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),
        #     #             "val_loss=", "{:.5f}".format(val_loss), "time=",
        #     #             "{:.5f}".format(time.time() - start))
        #     print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),
        #           "val_loss=", "{:.5f}".format(val_loss), "time=", "{:.5f}".format(time.time() - start))
        #
        # train_among_epochs.append(train_loss.avg)
        # val_among_epochs.append(val_loss)
        #
        # # print(int(val_loss.detach().cpu().numpy()))
        #
        # if (epoch < 30 and epoch > 10):
        #     if (len(set([round(val_e) for val_e in val_among_epochs[-20:]])) == 1):
        #         # stuck= True
        #         stop = False
        #         break
        #
        # if (epoch > args.early_stop):
        #     if (len(set([round(val_e) for val_e in val_among_epochs[-50:]])) == 1):  #
        #         print("break")
        #         # stop = True
        #         break
