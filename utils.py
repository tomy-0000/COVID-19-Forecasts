import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, xt):
        self.x = xt[0]
        self.t = xt[1]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.t[idx]).float()

    def __len__(self):
        return len(self.x)

class TrainValTest:
    def __init__(self, data, use_seq, predict_seq, batch_size=10000):
        train_data, val_data, test_data = data

        train_dataset = self._make_dataset(train_data, use_seq, predict_seq)
        val_dataset = self._make_dataset(val_data, use_seq, predict_seq)
        test_dataset = self._make_dataset(test_data, use_seq, predict_seq)
        self.dataset_dict = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        self.dataloader_dict = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}

    def _make_dataset(self, data, use_seq, predict_seq):
        a = predict_seq + use_seq - 1
        data_x, data_t = [], []
        for i in range(len(data) - a):
            data_x.append(data[i:i + use_seq])
            data_t.append(data[i + use_seq:i + use_seq + predict_seq])
        data_xt = Dataset([data_x, data_t])
        return data_xt

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = 1e10
        self.early_stop = False
        self.state_dict = None

    def __call__(self, net, score):
        if score <= self.best_score:
            self.best_score = score
            self.state_dict = net.state_dict()
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
        data[:, self.normalization_idx] = (data[:, self.normalization_idx] - self.mean)/self.std
        return data

    def inverse_standard(self, data):
        return data*self.std[0] + self.mean[0]
