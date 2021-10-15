import copy

import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, t):
        self.x = x
        self.t = t

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]).float(),
            torch.from_numpy(self.t[idx]).float(),
        )

    def __len__(self):
        return len(self.x)


class TrainValTest:
    def __init__(self, data, normalization_idx, use_seq, predict_seq, batch_size=10000):
        train_data, val_data, test_data = data
        self.data_dict = {"train": train_data, "val": val_data, "test": test_data}

        std = Standard(train_data, normalization_idx)
        train_data = std.standard(train_data)
        val_data = std.standard(val_data)
        test_data = std.standard(test_data)
        self.std = std

        train_dataset = self._make_dataset(train_data, use_seq, predict_seq)
        val_dataset = self._make_dataset(val_data, use_seq, predict_seq)
        test_dataset = self._make_dataset(test_data, use_seq, predict_seq)
        self.dataset_dict = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size
        )
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size
        )
        self.dataloader_dict = {
            "train": train_dataloader,
            "val": val_dataloader,
            "test": test_dataloader,
        }

    def _make_dataset(self, data, use_seq, predict_seq):
        x, t = [], []
        for i in range(len(data) - use_seq - predict_seq + 1):
            x.append(data[i : i + use_seq])
            t.append(data[i + use_seq : i + use_seq + predict_seq, 0])
        dataset = Dataset(x, t)
        return dataset


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_value = 1e10
        self.early_stop = False
        self.state_dict = None

    def __call__(self, net, value):
        if value <= self.best_value:
            self.best_value = value
            self.state_dict = copy.deepcopy(net.state_dict())
            self.counter = 0
        else:
            self.counter += 1
        if self.counter == self.patience:
            return True
        else:
            return False


class Standard:
    def __init__(self, data, normalization_idx):
        self.normalization_idx = normalization_idx
        self.mean = np.mean(data[:, normalization_idx], axis=0)
        self.std = np.std(data[:, normalization_idx], axis=0)

    def standard(self, data):
        data = data.copy()
        data[:, self.normalization_idx] = (
            data[:, self.normalization_idx] - self.mean
        ) / self.std
        return data

    def inverse_standard(self, data):
        return data * self.std[0] + self.mean[0]
