# coding: utf-8
import os
import time
import argparse
import networkx as nx
import numpy as np

from utils.tools import adjust_learning_rate
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from math import ceil
from tqdm import tqdm
import itertools
import pandas as pd
from model.mpnn_lstm import MPNNLSTM
# from utils.log import Log
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
from utils.metrics import AverageMeter
from model.MPNN_test import test_model
from model.A3TGCN import TemporalGNN
from model.MPNN_LSTM import MPNNLSTM
from utils.tools import EarlyStopping

writer = SummaryWriter('runs/scalar_example')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--model', type=str, default='MPNN_LSTM',
                    help='choose from MPNN_LSTM')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Size of batch.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate.')
parser.add_argument('--num_layers', type=int, default=1,
                    help='The num of Transformer Decoder layer')
parser.add_argument('--window', type=int, default=64,
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
parser.add_argument('--checkpoints', type=str, default='./res/',
                    help='location of model checkpoints')
parser.add_argument('--lradj', type=str, default='type2',
                    help='adjust learning rate')

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
    loss_train = F.mse_loss(output, y.squeeze(2))
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


def model_test(adj_edge_index, adj_edge_attr, features, y):
    output = model(features, adj_edge_index, adj_edge_attr)
    loss_test = F.mse_loss(output, y)
    return output, loss_test

def test(self, setting):
    test_data, test_loader = self._get_data(flag='test')

    self.model.eval()

    preds = []
    trues = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        pred, true = self._process_one_batch(
            test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
        preds.append(pred.detach().cpu().numpy())
        trues.append(true.detach().cpu().numpy())

    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print("mae:{:.4f} mse:{:.4f} rmse{:.4f} mape:{:.4f} mspe:{:.4f}".format(mae, mse, rmse, mape, mspe))
    # result save
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))

    np.save(folder_path + 'metrics3.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path + 'pred3.npy', preds)
    np.save(folder_path + 'true3.npy', trues)

    return

def predict(self, setting, load=False):
    pred_data, pred_loader = self._get_data(flag='pred')

    if load:
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

    self.model.eval()

    preds = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
        pred, true = self._process_one_batch(
            pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
        preds.append(pred.detach().cpu().numpy())

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    # result save
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(folder_path + 'real_prediction.npy', preds)

    return
if __name__ == "__main__":
    loader = AirDatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=args.window, num_timesteps_out=1)

    print("Dataset type:  ", dataset)
    print("Number of samples / sequences: ", len(set(dataset)))

    train_dataset, valid_dataset, test_dataset = temporal_signal_split_valid(dataset, ratio=[0.8, 0.1, 0.1])
    train_loader, valid_loader, test_loader = CreateDataloader(train_dataset, valid_dataset, test_dataset)
    os.chdir('../')
    os.chdir('../')
    for snapshot in train_dataset:
        static_edge_index = snapshot.edge_index.to(device)
        static_edge_attr = snapshot.edge_attr.to(device)
        break

    if args.model == "MPNN_LSTM":

        model = test_model(in_channels=args.node_features, hidden_size=args.hidden, out_channels=32, num_nodes=12,
                           window=args.window, dropout=args.dropout, num_layers=1, dec_seq_len=1,
                           batch_size=args.batch_size).to(device)
        # model = MPNNLSTM(nfeat=11, nhid=128, nout=1, nodes=12,window=12,
        #                  dropout=args.dropout,batch_size=args.batch_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        early_stopping = EarlyStopping(patience=7, verbose=True)
        path = os.path.join(args.checkpoints, 'run1')
        if not os.path.exists(path):
            os.makedirs(path)

        for epoch in tqdm(range(args.epochs), "training:"):
            start = time.time()
            model.train()
            train_loss = []
            # Train for one epoch y[256 12 1]
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
                val_loss = F.mse_loss(y_hat, y.squeeze(2))
                total_loss.append(val_loss.detach().cpu().item())

            print("Epoch {}: val MSE loss {:.4f}".format(epoch + 1, sum(total_loss) / len(train_loss)))
            writer.add_scalar('val_loss', sum(total_loss) / len(total_loss), epoch)
            vali_loss = sum(total_loss) / len(train_loss)
            early_stopping(vali_loss, model, path=path)
            adjust_learning_rate(optimizer, epoch + 1, args)

        best_model_path = path + '/' + 'checkpoint.pth'
        print("have save best model : {}".format(best_model_path))
        model.load_state_dict(torch.load(best_model_path))

        for batch, (X, y) in enumerate(test_loader):
            y_hat = model(X, static_edge_index, static_edge_attr)
            val_loss = F.mse_loss(y_hat, y.squeeze(2))

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