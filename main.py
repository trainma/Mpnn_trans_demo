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

#
now = datetime.datetime.now()
date_now = str(now)[:-7]
train_trans_log = Log("train", './log/' + date_now + '.txt')

# train_trans_log = Log("train", "./log/MPNN_trans.txt")
logger = train_trans_log.get_log()
writer = SummaryWriter('runs/scalar_example')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=120, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--model', type=str, default='test_model', help='choose from MPNN_LSTM')
parser.add_argument('--batch-size', type=int, default=16, help='Size of batch.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate.')
parser.add_argument('--num_layers', type=int, default=1, help='The num of Transformer Decoder layer')
parser.add_argument('--window', type=int, default=64, help='Size of window for features.')
parser.add_argument('--graph-window', type=int, default=7, help='Size of window for graphs in MPNN LSTM.')
parser.add_argument('--recur', default=False, help='True or False.')
parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle dataset')
parser.add_argument('--ratio', type=list, default=[0.8, 0.1, 0.1], help='the split ratio of train/valid/test')
parser.add_argument('--pred_len', type=int, default=8, help='the pred_len of the model')
parser.add_argument('--num_nodes', type=int, default=12, help='the num node of graph')
parser.add_argument('--node_features', type=int, default=9, help='the features of every node in graph')
parser.add_argument('--checkpoints', type=str, default='./res/', help='location of model checkpoints')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


def train(epoch, adj_edge_index, adj_edge_attr, features, y):
    optimizer.zero_grad()
    output = model(features, adj_edge_index, adj_edge_attr)
    loss_train = loss_fn(output.squeeze(2), y.squeeze(2))
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


def train2(epoch, adj_edge_index, adj_edge_attr, poi_edge_index, poi_edge_attr, ST_edge_index, ST_edge_attr, features,
           y):
    optimizer.zero_grad()
    output = model(features, adj_edge_index, adj_edge_attr, poi_edge_index, poi_edge_attr, ST_edge_index, ST_edge_attr)

    loss_train = loss_fn(output.squeeze(-1), y.transpose(1, 2))
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


def train3(epoch, adj_edge_index, adj_edge_attr, poi_edge_index, poi_edge_attr, ST_edge_index, ST_edge_attr, features,
           y):
    optimizer.zero_grad()
    output = model(features, adj_edge_index, adj_edge_attr, poi_edge_index, poi_edge_attr, ST_edge_index, ST_edge_attr)

    loss_train = loss_fn(output.squeeze(-1), y.squeeze(-1))
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


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
        = loader.get_dataset(num_timesteps_in=args.window, num_timesteps_out=8)
    print("Dataset type:  ", dataset)
    print("Number of samples / sequences: ", len(set(dataset)))

    train_dataset, valid_dataset, test_dataset = temporal_signal_split_valid(dataset, ratio=[0.8, 0.1, 0.1])
    train_loader, valid_loader, test_loader = CreateDataloader(train_dataset, valid_dataset, test_dataset, DEVICE=device
                                                               , batch_size=args.batch_size)
    os.chdir('../')
    os.chdir('../')
    print("Number of train_samples: ", len(set(train_dataset)))
    print("Number of valid_samples: ", len(set(valid_dataset)))
    print("number of test_dataset: ", len(set(test_dataset)))
    poi_graph_edge_index, poi_graph_edge_attr, ST_graph_edge_index, ST_graph_edge_attr, geograph_edge_index, geograph_edge_attr = get_edge_index_attr()

    if args.model == "test_model":
        model = Multi_graph_trans(in_channels=args.node_features, hidden_size=args.hidden, out_channels=32, d_model=256,
                                  feature_dim=9, label_len=24,
                                  num_nodes=12, window=args.window, dropout=args.dropout, num_layers=1, dec_seq_len=1,
                                  batch_size=args.batch_size).to(device)

        # model = test_model(in_channels=args.node_features, hidden_size=args.hidden, out_channels=32, num_nodes=12,
        #                    window=args.window, dropout=args.dropout, num_layers=1, dec_seq_len=1,
        #                    batch_size=args.batch_size).to(device)
        # model = MPNN_trans(nfeat=args.node_features, nhid=args.hidden, nout=1, nodes=12, window=args.window,
        #                    dropout=args.dropout,
        #                    batch_size=args.batch_size, label_len=48 ).to(device)
        # model = TemporalGNN(node_features=args.node_features,periods=1,batch_size=args.batch_size).to(device)
        # model = MPNNLSTM(nfeat=11, nhid=128, nout=1, nodes=12,window=12,
        #                  dropout=args.dropout,batch_size=args.batch_size).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        early_stopping = EarlyStopping(patience=7, verbose=True)
        path = os.path.join(args.checkpoints, 'run1')
        loss_fn = torch.nn.MSELoss()
        if not os.path.exists(path):
            os.makedirs(path)

        for epoch in tqdm(range(args.epochs), "training:"):
            start = time.time()
            model.train()
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
            for batch, (X, y) in enumerate(valid_loader):
                y_hat = model(X, geograph_edge_index, geograph_edge_attr, poi_graph_edge_index, poi_graph_edge_attr,
                              ST_graph_edge_index, ST_graph_edge_attr)
                val_loss = loss_fn(y_hat.squeeze(-1), y.transpose(1, 2))

                real = y.detach().cpu().numpy()
                # real = real.transpose(0, 2, 1)
                # real_data = label_scaler.inverse_transform(real.squeeze(2))
                real_data = label_scaler.inverse_transform(real.reshape(-1, 12))
                pred = y_hat.squeeze(-1).detach().cpu().numpy()
                # pred = np.expand_dims(pred, 1)
                pred_data = label_scaler.inverse_transform(pred.reshape(-1, 12))
                val_mae, val_mse, val_rmse, val_mape, val_mspe = metric(pred_data, real_data)
                mae_l.append(val_mae)
                mse_l.append(val_mse)
                rmse_l.append(val_rmse)
                mape_l.append(val_mape)
                mspe_l.append(val_mspe)
                total_loss.append(val_loss.detach().cpu().item())

            print("Epoch {}: val MSE loss {:.4f}".format(epoch + 1, sum(total_loss) / len(total_loss)))
            logger.info("Epoch {}: val MSE loss {:.4f}".format(epoch + 1, sum(total_loss) / len(total_loss)))
            metric_val = "Epoch {}: val_mae {:.4f} val_mse {:.4f} val_rmse {:.4f} val_mape {:.4f} val_mspe {:.4f}".format(
                epoch + 1, sum(mae_l) / len(mae_l), sum(mse_l) / len(mse_l), sum(rmse_l) / len(rmse_l),
                sum(mape_l) / len(mape_l), sum(mspe_l) / len(mspe_l))
            print(metric_val)
            logger.info(metric_val)
            writer.add_scalar('val_loss', sum(total_loss) / len(total_loss), epoch)
            vali_loss = sum(total_loss) / len(total_loss)
            early_stopping(vali_loss, model, path=path)
            adjust_learning_rate(optimizer, epoch + 1, args)

            best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
        print("have save best model : {}".format(best_model_path))
        # for batch, (X, y) in enumerate(test_loader):
        #     y_hat = model(X, static_edge_index, static_edge_attr)
        #     val_loss = F.mse_loss(y_hat, y.squeeze(2))

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
