import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric_temporal import StaticGraphTemporalSignal


class BaseLineDatasetLoader(object):
    def __init__(self, args):
        super(BaseLineDatasetLoader, self).__init__()
        self.raw_data = pd.read_csv('data/Airquality/Aggr_PM2.5.csv')

    def _get_dataset_multi_step(self, Source, scope, train_test_split, window_size, n_out):
        split = int(train_test_split * scope)
        df = Source
        scaler = MinMaxScaler()
        scaler2 = MinMaxScaler()
        open_arr = scaler.fit_transform(df.values)
        raw_labels = df['var1'].values.reshape(-1, 1)
        label_list = np.zeros(shape=(scope - window_size - n_out, n_out))

        for i in range(scope - window_size - n_out):
            label_list[i, :] = raw_labels[i + window_size:i + window_size + n_out, 0]

        label_normalized = scaler2.fit_transform(label_list)

        X = np.zeros(shape=(scope - window_size - n_out, window_size, open_arr.shape[1]))

        for i in range(scope - window_size - n_out):
            X[i, :] = open_arr[i:i + window_size, :]

        train_X = X[:split, :]
        train_label = label_normalized[:split, :]
        test_X = X[split:scope, :]
        test_label = label_normalized[split:scope, :]
        return train_X, train_label, test_X, test_label, scaler, scaler2

    def _loader_dataset(self):
        train_X, train_label, test_X, test_label, scaler, scaler2 = self._get_dataset_multi_step(self.raw_data, )


class BaselineDatasetLoader2(object):
    def __init__(self):
        super(BaselineDatasetLoader2, self).__init__()
        os.chdir("./data")
        self.meta_labs = {}
        self.meta_features = []
        self.meta_y = []
        self.meta_graph = None
        self.data_scaler = None
        self.meta_data = None

    def _process_dataset(self):
        os.chdir("Airquality")

        for file in os.listdir('./labels'):
            labels = pd.read_csv('./labels/' + file)
            labels = labels.drop('station', axis=1)
            labels = labels.drop('date', axis=1)
            labels.drop('Unnamed: 0', axis=1, inplace=True)
            # labels = labels.set_index("dates")
            self.meta_labs[file.split('_')[0]] = labels['PM2.5']

        edge_index = pd.read_csv('./geo_graph.csv', header=None).iloc[:, [0, 1]].values
        edge_attr = pd.read_csv('./geo_graph.csv', header=None).iloc[:, -1].values
        edge_attr = torch.from_numpy(edge_attr)
        edge_index = torch.from_numpy(edge_index)
        self.meta_graph = Data(edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)

        with open('./station.json', 'r') as fp:
            stations = json.load(fp)
        addr = list(stations)
        df = self.meta_labs[addr[0]]

        for i in addr:
            if i == addr[0]:
                continue
            else:
                df = pd.concat((df, self.meta_labs[i]), axis=1)

        raw_data = df.values
        self.data_scaler = StandardScaler().fit(raw_data)
        self.normalize_data = self.data_scaler.transform(raw_data)
        # data = np.zeros((12, normalize_data.shape[0], 11))  # [nodes,features,time_stamps]
        #
        # for i in range(data.shape[0]):
        #     if i == 0:
        #         data[i] = normalize_data[:, :11]
        #     else:
        #         data[i] = normalize_data[:, 11 * i:11 * i + 11]
        #
        # data = data.transpose(0, 2, 1)
        # self.meta_data = data

    def _get_edges_and_weights(self):
        self.edges = self.meta_graph.edge_index.numpy()
        self.edge_weights = self.meta_graph.edge_attr.numpy()

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):

        # indices = [
        #     (i, i + (num_timesteps_in + num_timesteps_out))
        #     for i in range(self.meta_data.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        # ]
        # features, target = [], []
        # for i, j in indices:
        #     features.append((self.meta_data[:, :, i: i + num_timesteps_in]))
        #     target.append((self.meta_data[0, 0, i + num_timesteps_in: j]))
        # # self.features [207(nodes) 2(feature_dim) 12(timesteps)]
        # # self.target [207(nodes) 12(timesteps)]
        # self.features = features
        # self.targets = target

        stacked_target = self.normalize_data
        self.features = [
            stacked_target[i: i + num_timesteps_in, :].T
            for i in range(stacked_target.shape[0] - num_timesteps_in - num_timesteps_out + 1)
        ]
        self.targets = [
            stacked_target[i + num_timesteps_in:i + num_timesteps_in + num_timesteps_out, :].T
            for i in range(stacked_target.shape[0] - num_timesteps_in - num_timesteps_out + 1)
        ]

    def get_dataset(
            self, num_timesteps_in: int = 72, num_timesteps_out: int = 12
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for Beijing_air_quality dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Beijing_air_quality dataset
                forecasting dataset.
        """
        self._process_dataset()
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset
