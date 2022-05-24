import argparse
import os

import numpy as np
import torch

from dataset.Airquality_dataset import AirDatasetLoader
from dataset.custom_dataloader import CreateDataloader
from dataset.temporal_split import temporal_signal_split_valid
from model.Multi_graph_trans import Multi_graph_trans
from utils.metrics import metric

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=120,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--model', type=str, default='test_model',
                    help='choose from MPNN_LSTM')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Size of batch.')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate.')
parser.add_argument('--num_layers', type=int, default=1,
                    help='The num of Transformer Decoder layer')
parser.add_argument('--window', type=int, default=64,  # (48h 2day to predict one hour)
                    help='Size of window for features.')
parser.add_argument('--graph-window', type=int, default=7,
                    help='Size of window for graphs in MPNN LSTM.')
parser.add_argument('--recur', default=False,
                    help='True or False.')
parser.add_argument('--sep', type=int, default=10,
                    help='Seperator for validation and train set.')
parser.add_argument('--shuffle', type=bool, default=False,
                    help='whether to shuffle dataset')
parser.add_argument('--ratio', type=list, default=[0.8, 0.1, 0.1],
                    help='the split ratio of train/valid/test')
parser.add_argument('--pred_len', type=int, default=1,
                    help='the pred_len of the model')
parser.add_argument('--num_nodes', type=int, default=12,
                    help='the num node of graph')
parser.add_argument('--node_features', type=int, default=9,
                    help='the features of every node in graph')
parser.add_argument('--checkpoints', type=str, default='./res/',
                    help='location of model checkpoints')
parser.add_argument('--lradj', type=str, default='type2',
                    help='adjust learning rate')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


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


if __name__ == '__main__':
    model = Multi_graph_trans(in_channels=args.node_features, hidden_size=args.hidden, out_channels=32, d_model=256,
                              feature_dim=9,
                              num_nodes=12, window=args.window, dropout=args.dropout, num_layers=1, dec_seq_len=1,
                              batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load('./res/rmse_11.pth'))
    print(model)
    loader = AirDatasetLoader()
    dataset, _, label_scaler, poi_graph, ST_graph \
        = loader.get_dataset(num_timesteps_in=args.window, num_timesteps_out=1)
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
    loss_fn = torch.nn.MSELoss()
    mae_l, mse_l, rmse_l, mape_l, mspe_l = [], [], [], [], []
    total_loss = []
    model.eval()
    pred_data_list = []
    real_data_list = []
    for batch, (X, y) in enumerate(test_loader):
        y_hat = model(X, geograph_edge_index, geograph_edge_attr, poi_graph_edge_index, poi_graph_edge_attr,
                      ST_graph_edge_index, ST_graph_edge_attr)
        val_loss = loss_fn(y_hat.squeeze(2), y.squeeze(2))
        real = y.detach().cpu().numpy()
        # real = real.transpose(0, 2, 1)
        real_data = label_scaler.inverse_transform(real.squeeze(2))
        real_data_list.append(real_data)
        pred = y_hat.squeeze(2).detach().cpu().numpy()
        # pred = np.expand_dims(pred, 1)
        pred_data = label_scaler.inverse_transform(pred)
        pred_data_list.append(pred_data)
        val_mae, val_mse, val_rmse, val_mape, val_mspe = metric(pred_data, real_data)
        mae_l.append(val_mae)
        mse_l.append(val_mse)
        rmse_l.append(val_rmse)
        mape_l.append(val_mape)
        mspe_l.append(val_mspe)
        total_loss.append(val_loss.detach().cpu().item())
    print("val MSE loss {:.4f}".format(sum(total_loss) / len(total_loss)))
    metric_val = "val_mae {:.4f} val_mse {:.4f} val_rmse {:.4f} val_mape {:.4f} val_mspe {:.4f}".format(
        sum(mae_l) / len(mae_l), sum(mse_l) / len(mse_l), sum(rmse_l) / len(rmse_l),
        sum(mape_l) / len(mape_l), sum(mspe_l) / len(mspe_l))
    print(metric_val)

    real_datas = np.concatenate(real_data_list)
    pred_datas = np.concatenate(pred_data_list)

    np.save('visual/pred2-val.npy', pred_datas)
    np.save('visual/real2-val.npy', real_datas)
    print("have save done !")
