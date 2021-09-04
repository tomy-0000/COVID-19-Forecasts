import numpy as np
import pandas as pd
import torch.nn as nn
from utils import Standard

class Net(nn.Module):
    def __init__(self, hidden_size, num_layers, predict_seq):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, predict_seq)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y

    @staticmethod
    def get_data(i, use_seq, predict_seq):
        data = pd.read_csv("data_use/count.csv")[["東京都"]]
        data = data.values.astype(float)
        total_seq = use_seq + predict_seq
        train_data = data[:-2*total_seq]
        val_data = data[-2*total_seq:-total_seq]
        test_data = data[-total_seq:]
        return [train_data, val_data, test_data]

    normalization_idx = [0]
    net_params = {
        # "hidden_size": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        "hidden_size": [8],
        # "num_layers": [1, 2]
        "num_layers": [1]
    }

# 特徴量
#   カウント
