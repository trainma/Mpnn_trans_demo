import os
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
import pandas as pd
from utils.log import Log

from torch_geometric.data import Data
from torch_geometric_temporal.nn.recurrent import MPNNLSTM
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal import StaticGraphTemporalSignal
import json
from sklearn.preprocessing import StandardScaler
from dataset.temporal_split import temporal_signal_split_valid

class AirDatasetLoader(object):
    def __init__(self):
        super(AirDatasetLoader, self).__init__()
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
            self.meta_labs[file.split('_')[0]] = labels

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
        normalize_data = self.data_scaler.transform(raw_data)
        data = np.zeros((12, normalize_data.shape[0], 11))  # [nodes,features,time_stamps]

        for i in range(data.shape[0]):
            if i == 0:
                data[i] = normalize_data[:, :11]
            else:
                data[i] = normalize_data[:, 11 * i:11 * i + 11]

        data = data.transpose(0, 2, 1)
        self.meta_data = data

    def _get_edges_and_weights(self):
        self.edges = self.meta_graph.edge_index.numpy()
        self.edge_weights = self.meta_graph.edge_attr.numpy()

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):

        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.meta_data.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]
        features, target = [], []
        for i, j in indices:
            features.append((self.meta_data[:, :, i: i + num_timesteps_in]))
            target.append((self.meta_data[:, 0, i + num_timesteps_in: j]))
        # self.features [207(nodes) 2(feature_dim) 12(timesteps)]
        # self.target [207(nodes) 12(timesteps)]
        self.features = features
        self.targets = target

    def get_dataset(
            self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
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

class AirDatasetLoader2(object):
    def __init__(self):
        super(AirDatasetLoader2, self).__init__()
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
            self.meta_labs[file.split('_')[0]] = labels

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
        normalize_data = self.data_scaler.transform(raw_data)
        data = np.zeros((12, normalize_data.shape[0], 11))  # [nodes,features,time_stamps]

        for i in range(data.shape[0]):
            if i == 0:
                data[i] = normalize_data[:, :11]
            else:
                data[i] = normalize_data[:, 11 * i:11 * i + 11]

        data = data.transpose(0, 2, 1)
        self.meta_data = data

    def _get_edges_and_weights(self):
        self.edges = self.meta_graph.edge_index.numpy()
        self.edge_weights = self.meta_graph.edge_attr.numpy()

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):

        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.meta_data.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]
        features, target = [], []
        for i, j in indices:
            features.append((self.meta_data[:, :, i: i + num_timesteps_in]))
            target.append((self.meta_data[0, 0, i + num_timesteps_in: j]))
        # self.features [207(nodes) 2(feature_dim) 12(timesteps)]
        # self.target [207(nodes) 12(timesteps)]
        self.features = features
        self.targets = target

    def get_dataset(
            self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
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
if __name__ == "__main__":
    loader = AirDatasetLoader2()
    dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=1)
    print(dataset)
