# coding: utf-8
import os
import time
import argparse
import networkx as nx
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from tqdm import tqdm
import itertools
import pandas as pd

from model.mpnn_lstm import MPNNLSTM
from utils.log import Log

train_trans_log = Log("train", "./log/MPNN_trans.txt")
logger = train_trans_log.get_log()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from dataset.Airquality_dataset import AirDatasetLoader
from dataset.temporal_split import temporal_signal_split_valid

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--batch-size', type=int, default=16,
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
parser.add_argument('--shuffle', type=bool, default=True,
                    help='whether to shuffle dataset')
parser.add_argument('--ratio', type=list, default=[0.8, 0.1, 0.1],
                    help='the split ratio of train/valid/test')
parser.add_argument('--pred_len', type=int, default=12,
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


if __name__ == "__main__":
    loader = AirDatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=args.window, num_timesteps_out=args.pred_len)

    print("Dataset type:  ", dataset)
    print("Number of samples / sequences: ", len(set(dataset)))

    train_dataset, valid_dataset, test_dataset = temporal_signal_split_valid(dataset, ratio=[0.8, 0.1, 0.1])
    train_loader, valid_loader, test_loader = CreateDataloader(train_dataset, valid_dataset, test_dataset)
