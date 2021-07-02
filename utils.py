import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
if "get_ipython" in globals():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
sns.set(font="IPAexGothic")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, use_seq, predict_seq):
        self.use_seq = use_seq
        self.predict_seq = predict_seq
        self.data = torch.from_numpy(data).float()

    def __getitem__(self, idx):
        return (self.data[idx:idx + self.use_seq],
            self.data[idx + self.use_seq:idx + self.use_seq + self.predict_seq, [0]])

    def __len__(self):
        return len(self.data) - self.use_seq - self.predict_seq + 1

    def get_all_label(self):
        return self.data[:, 0].numpy()


class TrainValTest:
    def __init__(self, data, normalization_idx):
        self.seq = 30
        self.predict_seq = 30
        self.batch_size = 10000
        data = data.copy()
        val_test_len = self.seq + self.predict_seq
        train_data = data[:-2*val_test_len]
        val_data = data[-2*val_test_len:-val_test_len]
        test_data = data[-val_test_len:]

        self.normalization_idx = normalization_idx
        self.mean = np.mean(train_data[:, normalization_idx], axis=0)
        self.std = np.std(train_data[:, normalization_idx], axis=0)
        train_data[:, normalization_idx] = (train_data[:, normalization_idx] - self.mean)/self.std
        val_data[:, normalization_idx] = (val_data[:, normalization_idx] - self.mean)/self.std
        test_data[:, normalization_idx] = (test_data[:, normalization_idx] - self.mean)/self.std

        train_dataset = Dataset(train_data, self.seq, self.predict_seq)
        val_dataset = Dataset(val_data, self.seq, self.predict_seq)
        test_dataset = Dataset(test_data, self.seq, self.predict_seq)
        self.dataset_dict = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)
        self.dataloader_dict = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}

    def inverse_standard(self, data):
        return data*self.std[0] + self.mean[0]

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
