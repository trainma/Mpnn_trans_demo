# coding: utf-8
import argparse
import datetime
import os
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.Airquality_dataset import AirDatasetLoader
from dataset.custom_dataloader import CreateDataloader
from dataset.temporal_split import temporal_signal_split_valid
from model.Multi_graph_trans import Multi_graph_trans
from utils.log import Log
from utils.metrics import metric
from utils.tools import EarlyStopping
from utils.tools import adjust_learning_rate
from Exp import Exp

#
now = datetime.datetime.now()
date_now = str(now)[:-7]
train_trans_log = Log("train", './log/' + date_now + '.txt')

# train_trans_log = Log("train", "./log/MPNN_trans.txt")
logger = train_trans_log.get_log()
writer = SummaryWriter('runs/scalar_example')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=72, help='Number of hidden units.')
parser.add_argument('--batch-size', type=int, default=64, help='Size of batch.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate.')
parser.add_argument('--num_layers', type=int, default=1, help='The num of Transformer Decoder layer')
parser.add_argument('--window', type=int, default=72, help='Size of window for features.')
parser.add_argument('--recur', default=False, help='True or False.')
parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle dataset')
parser.add_argument('--ratio', type=list, default=[0.8, 0.1, 0.1], help='the split ratio of train/valid/test')
parser.add_argument('--pred_len', type=int, default=24, help='the pred_len of the model')
parser.add_argument('--num_nodes', type=int, default=12, help='the num node of graph')
parser.add_argument('--node_features', type=int, default=9, help='the features of every node in graph')
parser.add_argument('--checkpoints', type=str, default='./res/', help='location of model checkpoints')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--model', type=str, default='A3TGCN', help='choose what model to train')
parser.add_argument('--d_model', type=int, default=256, help='the d_model of the transformer')
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--enc_layers', type=int, default=1, help='the number of enc layers in trans')
parser.add_argument('--dec_layers', type=int, default=1, help='the number of dec layers in trans')
parser.add_argument('--num_workers', type=int, default=0)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


# def train(epoch, adj_edge_index, adj_edge_attr, features, y):
#     optimizer.zero_grad()
#     output = model(features, adj_edge_index, adj_edge_attr)
#     loss_train = loss_fn(output.squeeze(2), y.squeeze(2))
#     loss_train.backward(retain_graph=True)
#     optimizer.step()
#     return output, loss_train
#
#
# def train2(epoch, adj_edge_index, adj_edge_attr, poi_edge_index, poi_edge_attr, ST_edge_index, ST_edge_attr, features,
#            y):
#     optimizer.zero_grad()
#     output = model(features, adj_edge_index, adj_edge_attr, poi_edge_index, poi_edge_attr, ST_edge_index, ST_edge_attr)
#
#     loss_train = loss_fn(output.squeeze(-1), y.transpose(1, 2))
#     loss_train.backward(retain_graph=True)
#     optimizer.step()
#     return output, loss_train
#
#
# def train3(epoch, adj_edge_index, adj_edge_attr, poi_edge_index, poi_edge_attr, ST_edge_index, ST_edge_attr, features,
#            y):
#     optimizer.zero_grad()
#     output = model(features, adj_edge_index, adj_edge_attr, poi_edge_index, poi_edge_attr, ST_edge_index, ST_edge_attr)
#
#     loss_train = loss_fn(output.squeeze(-1), y.squeeze(-1))
#     loss_train.backward(retain_graph=True)
#     optimizer.step()
#     return output, loss_train


def get_edge_index_attr():
    poi_graph_edge_index, poi_graph_edge_attr = poi_graph.edge_index.to(device), poi_graph.edge_attr.to(device,
                                                                                                        torch.float32)
    ST_graph_edge_index, ST_graph_edge_attr = ST_graph.edge_index.to(device), ST_graph.edge_attr.to(device,
                                                                                                    torch.float32)
    for snapshot in train_dataset:
        geograph_edge_index = snapshot.edge_index.to(device)
        geograph_edge_attr = snapshot.edge_attr.to(device)
        break

    return poi_graph_edge_index, poi_graph_edge_attr, \
           ST_graph_edge_index, ST_graph_edge_attr, \
           geograph_edge_index, geograph_edge_attr


if __name__ == "__main__":
    loader = AirDatasetLoader()
    dataset, _, label_scaler, poi_graph, ST_graph \
        = loader.get_dataset(num_timesteps_in=args.window, num_timesteps_out=args.pred_len)
    print("Dataset type:  ", dataset)
    print("Number of samples / sequences: ", len(set(dataset)))

    train_dataset, valid_dataset, test_dataset = temporal_signal_split_valid(dataset, ratio=[0.8, 0.1, 0.1])
    train_loader, valid_loader, test_loader = CreateDataloader(train_dataset, valid_dataset, test_dataset, DEVICE=device
                                                               , batch_size=args.batch_size,
                                                               num_workers=args.num_workers)
    os.chdir('../')
    os.chdir('../')
    print("Number of train_samples: ", len(set(train_dataset)))
    print("Number of valid_samples: ", len(set(valid_dataset)))
    print("number of test_dataset: ", len(set(test_dataset)))
    poi_graph_edge_index, poi_graph_edge_attr, ST_graph_edge_index, ST_graph_edge_attr, geograph_edge_index, geograph_edge_attr = get_edge_index_attr()

    Experiment = Exp(args=args, device=device, label_scaler=label_scaler, train_loader=train_loader,
                     valid_loader=valid_loader, test_loader=test_loader, poi_graph_edge_index=poi_graph_edge_index,
                     poi_graph_edge_attr=poi_graph_edge_attr, ST_graph_edge_index=ST_graph_edge_index,
                     ST_graph_edge_attr=ST_graph_edge_attr, geograph_edge_index=geograph_edge_index,
                     geograph_edge_attr=geograph_edge_attr)

    Experiment()
