import numpy as np
import torch
from torch.utils.data import DataLoader

def CreateDataloader(TrainDataset, ValidDataset, TestDataset, DEVICE,
                     batch_size, num_workers):
    def temp_dataloader(temp_dataset, shuffle):
        temp_input = np.array(temp_dataset.features)  # (27399, 207, 2, 12)
        temp_target = np.array(temp_dataset.targets)  # (27399, 207, 12)
        temp_x_tensor = torch.from_numpy(temp_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
        temp_target_tensor = torch.from_numpy(temp_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
        temp_dataset_new = torch.utils.data.TensorDataset(temp_x_tensor, temp_target_tensor)
        temp_loader = torch.utils.data.DataLoader(temp_dataset_new, batch_size=batch_size, shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  drop_last=True)
        return temp_loader

    train_loader = temp_dataloader(TrainDataset, True)
    valid_loader = temp_dataloader(ValidDataset, False)
    test_loader = temp_dataloader(TestDataset, False)
    return train_loader, valid_loader, test_loader
