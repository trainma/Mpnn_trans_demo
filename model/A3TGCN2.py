import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from dataset.Airquality_dataset import AirDatasetLoader
from dataset.temporal_split import temporal_signal_split_valid

# GPU support
DEVICE = torch.device('cuda')  # cuda
shuffle = True
batch_size = 64

# Dataset
# Traffic forecasting dataset based on Los Angeles Metropolitan traffic
# 207 loop detectors on highways
# March 2012 - June 2012
# From the paper: Diffusion Convolutional Recurrent Neural Network


# from torch_geometric_temporal.dataset import METRLADatasetLoader


# Visualize traffic over time
# sensor_number = 1
# hours = 24
# sensor_labels = [bucket.y[sensor_number][0].item() for bucket in list(dataset)[:hours]]
# plt.plot(sensor_labels)
# plt.show()
# plt.savefig('./label.png')

# Train test split

from torch_geometric_temporal.signal import temporal_signal_split


# Creating Dataloaders
def CreateDataloader(TrainDataset, ValidDataset, TestDataset):
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


# Making the model

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size, out_channels
                 ):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=out_channels, periods=periods,
                            batch_size=batch_size)  # node_features=2, periods=12
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels, periods)

    def forward(self, x, edge_index, edge_weight):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        # x.shape = (batch_size,node_num,feature_size,time_steps)
        h = self.tgnn(x, edge_index, edge_weight)  # x [b, 207, 2, 12]  returns h [b, 207, 12]
        h = F.relu(h)
        h = self.linear(h)
        return h


if __name__ == "__main__":
    loader = AirDatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=8)[0]
    print("Dataset type:  ", dataset)
    print("Number of samples / sequences: ", len(set(dataset)))

    train_dataset, valid_dataset, test_dataset = temporal_signal_split_valid(dataset, ratio=[0.8, 0.1, 0.1])

    print("Number of train buckets: ", len(set(train_dataset)))
    print("Number of Valid buckets: ", len(set(valid_dataset)))
    print("Number of test buckets: ", len(set(test_dataset)))
    train_loader, valid_loader, test_loader = CreateDataloader(train_dataset, valid_dataset, test_dataset)
    # Create model and optimizers
    model = TemporalGNN(node_features=9, periods=8, batch_size=batch_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)
    # --------------------------------------------------
    print('Optimizer\'s state_dict:')  # If you notice here the Attention is a trainable parameter
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    # Loading the graph once because it's a static graph

    for snapshot in train_dataset:
        stMPNN_transatic_edge_index = snapshot.edge_index.to(DEVICE)
        static_edge_attr = snapshot.edge_attr.to(DEVICE)
        break;

    # Training the model
    model.train()

    for epoch in range(100):
        step = 0
        loss_list = []
        for encoder_inputs, labels in train_loader:
            y_hat = model(encoder_inputs, static_edge_index, static_edge_attr)  # Get model predictions
            loss = loss_fn(y_hat,
                           labels)  # Mean squared error #loss = torch.mean((y_hat-labels)**2)  sqrt to change it to rmse
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step = step + 1
            loss_list.append(loss.item())
            if step % 100 == 0:
                print(sum(loss_list) / len(loss_list))
        print("Epoch {} train RMSE: {:.4f}".format(epoch, sum(loss_list) / len(loss_list)))

    ## Evaluation

    # - Lets get some sample predictions for a specific horizon (e.g. 288/12 = 24 hours)
    # - The model always gets one hour and needs to predict the next hour

    model.eval()
    step = 0
    # Store for analysis
    total_loss = []
    for encoder_inputs, labels in test_loader:
        # Get model predictions
        y_hat = model(encoder_inputs, static_edge_index, static_edge_attr)
        # Mean squared error
        loss = loss_fn(y_hat, labels)
        total_loss.append(loss.item())
        # Store for analysis below
        # test_labels.append(labels)
        # predictions.append(y_hat)

    print("Test MSE: {:.4f}".format(sum(total_loss) / len(total_loss)))

    ### Visualization

    # - The further away the point in time is, the worse the predictions get
    # - Predictions shape: [num_data_points, num_sensors, num_timesteps]

    # sensor = 0
    # timestep = 11
    # preds = np.asarray([pred[sensor][timestep].detach().cpu().numpy() for pred in y_hat])
    # labs = np.asarray([label[sensor][timestep].cpu().numpy() for label in labels])
    # print("Data points:,", preds.shape)
    #
    # plt.figure(figsize=(20, 5))
    # sns.lineplot(data=preds, label="pred")
    # sns.lineplot(data=labs, label="true")
    # plt.savefig('./pred.png')
