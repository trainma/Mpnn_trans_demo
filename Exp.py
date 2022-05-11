import argparse
import datetime
import os
import time
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.Multi_graph_trans import Multi_graph_trans
from model.A3TGCN2 import TemporalGNN
import torch.optim as optim
from utils.tools import EarlyStopping
from utils.tools import adjust_learning_rate
from utils.log import Log
from utils.metrics import metric

now = datetime.datetime.now()
date_now = str(now)[:-7]
Exp_log = Log("train", './log/' + date_now + '.txt')
logger = Exp_log.get_log()

writer = SummaryWriter('runs/scalar_example')


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


def train3(epoch, adj_edge_index, adj_edge_attr, poi_edge_index, poi_edge_attr, ST_edge_index, ST_edge_attr, features,
           y, optimizer: optim, model, loss_fn):
    optimizer.zero_grad()
    output = model(features, adj_edge_index, adj_edge_attr, poi_edge_index, poi_edge_attr, ST_edge_index, ST_edge_attr)

    loss_train = loss_fn(output.squeeze(-1), y.squeeze(-1))
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


def train_A3TGCN(epoch, adj_edge_index, adj_edge_attr, features,
                 y, optimizer: optim, model, loss_fn):
    optimizer.zero_grad()
    output = model(features, adj_edge_index, adj_edge_attr)

    loss_train = loss_fn(output, y)
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


class Exp():
    def __init__(self, args, device, label_scaler, train_loader, valid_loader, test_loader,
                 poi_graph_edge_index=None, poi_graph_edge_attr=None,
                 ST_graph_edge_index=None,
                 ST_graph_edge_attr=None, geograph_edge_index=None, geograph_edge_attr=None):
        self.model = None
        self.args = args
        self.device = device
        self.poi_graph_edge_index = poi_graph_edge_index
        self.poi_graph_edge_attr = poi_graph_edge_attr
        self.ST_graph_edge_index = ST_graph_edge_index
        self.ST_graph_edge_attr = ST_graph_edge_attr
        self.geograph_edge_index = geograph_edge_index
        self.geograph_edge_attr = geograph_edge_attr
        self.label_scaler = label_scaler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def _build_model(self):
        if self.args.model == "MPNN_trans":
            self.model = Multi_graph_trans(in_channels=self.args.node_features, hidden_size=self.args.hidden,
                                           out_channels=32, pred_len=self.args.pred_len,
                                           d_model=self.args.d_model,
                                           feature_dim=9, label_len=self.args.label_len,
                                           num_nodes=12, window=self.args.window, dropout=self.args.dropout,
                                           num_layers=self.args.num_layers,
                                           dec_seq_len=self.args.dec_layers,
                                           batch_size=self.args.batch_size).to(self.device)

        elif self.args.model == "A3TGCN":
            self.model = TemporalGNN(node_features=self.args.node_features
                                     , periods=self.args.pred_len
                                     , batch_size=self.args.batch_size, out_channels=self.args.hidden).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-3)
        self.early_stopping = EarlyStopping(patience=7, verbose=True)
        self.path = os.path.join(self.args.checkpoints, 'run1')
        self.loss_fn = torch.nn.MSELoss()
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def _train(self, epoch, train_loader):

        self.model.train()
        train_loss = []
        # Train for one epoch y[256 12 1]
        for batch, (X, y) in enumerate(train_loader):
            if self.args.model == 'MPNN_trans':
                output, loss = train2(epoch, self.geograph_edge_index, self.geograph_edge_attr,
                                      self.poi_graph_edge_index,
                                      self.poi_graph_edge_attr, self.ST_graph_edge_index, self.ST_graph_edge_attr, X, y,
                                      optimizer=self.optimizer,
                                      loss_fn=self.loss_fn, model=self.model)

            if self.args.model == "A3TGCN":
                output, loss = train_A3TGCN(epoch, self.geograph_edge_index, self.geograph_edge_attr, X, y,
                                            optimizer=self.optimizer, loss_fn=self.loss_fn, model=self.model)

            train_loss.append(loss.data.item())

        print("Epoch {}: train MSE loss {:.4f}".format(epoch + 1, sum(train_loss) / len(train_loss)))
        logger.info(
            self.args.model + " Epoch {}: train MSE loss {:.4f}".format(epoch + 1, sum(train_loss) / len(train_loss)))
        writer.add_scalar('train_loss', sum(train_loss) / len(train_loss), epoch)
        # Evaluate on validation set

    def _valid(self, epoch, valid_loader):
        if self.args.model == 'MPNN_trans':
            self.model.eval()
            total_loss = []
            mae_l, mse_l, rmse_l, mape_l, mspe_l = [], [], [], [], []
            for batch, (X, y) in enumerate(valid_loader):
                y_hat = self.model(X, self.geograph_edge_index, self.geograph_edge_attr, self.poi_graph_edge_index,
                                   self.poi_graph_edge_attr,
                                   self.ST_graph_edge_index, self.ST_graph_edge_attr)
                val_loss = self.loss_fn(y_hat.squeeze(-1), y.transpose(1, 2))

                real = y.detach().cpu().numpy()
                # real = real.transpose(0, 2, 1)
                # real_data = label_scaler.inverse_transform(real.squeeze(2))
                real_data = self.label_scaler.inverse_transform(real.reshape(-1, 12))
                pred = y_hat.squeeze(-1).detach().cpu().numpy()
                # pred = np.expand_dims(pred, 1)
                pred_data = self.label_scaler.inverse_transform(pred.reshape(-1, 12))
                val_mae, val_mse, val_rmse, val_mape, val_mspe = metric(pred_data, real_data)
                mae_l.append(val_mae)
                mse_l.append(val_mse)
                rmse_l.append(val_rmse)
                mape_l.append(val_mape)
                mspe_l.append(val_mspe)
                total_loss.append(val_loss.detach().cpu().item())

        elif self.args.model == 'A3TGCN':
            self.model.eval()
            total_loss = []
            mae_l, mse_l, rmse_l, mape_l, mspe_l = [], [], [], [], []
            for batch, (X, y) in enumerate(valid_loader):
                y_hat = self.output = self.model(X, self.geograph_edge_index, self.geograph_edge_attr)
                val_loss = self.loss_fn(y_hat, y)

                real = y.detach().cpu().numpy()
                # real = real.transpose(0, 2, 1)
                # real_data = label_scaler.inverse_transform(real.squeeze(2))
                real_data = self.label_scaler.inverse_transform(real.transpose(0, 2, 1).reshape(-1, 12))
                pred = y_hat.squeeze(-1).detach().cpu().numpy()
                # pred = np.expand_dims(pred, 1)
                pred_data = self.label_scaler.inverse_transform(pred.transpose(0, 2, 1).reshape(-1, 12))
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
        self.early_stopping(vali_loss, self.model, path=self.path)
        adjust_learning_rate(self.optimizer, epoch + 1, self.args)

    def _test(self):
        pass

    def __call__(self):
        self._build_model()
        for epoch in tqdm(range(self.args.epochs), "training:"):
            start = time.time()
            self._train(epoch, self.train_loader)
            self._valid(epoch, self.valid_loader)
            end = time.time()

            best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        print("have save best model : {}".format(best_model_path))
