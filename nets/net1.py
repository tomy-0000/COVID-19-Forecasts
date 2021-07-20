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

    @classmethod
    def get_data(cls, i, use_seq, predict_seq):
        df = pd.read_csv("./data/count_tokyo.csv", parse_dates=True, index_col=0)
        data = df.values.astype(float)[150:-i*predict_seq]
        train_data = data[:-2*(use_seq + predict_seq)]
        val_data = data[-2*(use_seq + predict_seq):-(use_seq + predict_seq)]
        test_data = data[-(use_seq + predict_seq):]

        std = Standard(train_data, cls.normalization_idx)
        train_data = std.standard(train_data)
        val_data = std.standard(val_data)
        test_data = std.standard(test_data)
        return [train_data, val_data, test_data], std

    normalization_idx = [0]
    net_params = [
        ("hidden_size", [1, 2, 4, 8, 16, 32, 64, 128, 256]),
        ("num_layers", [1, 2])
    ]

# 特徴量
#   カウント
