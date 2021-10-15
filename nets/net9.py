import pandas as pd
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, hidden_size, num_layers, predict_seq):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, predict_seq)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y

    @staticmethod
    def get_data():
        df1 = pd.read_csv("./data/count_tokyo.csv", parse_dates=True, index_col=0)
        df2 = pd.read_csv("./data/emergency.csv", parse_dates=True, index_col=0)
        df = pd.concat([df1, df2], axis=1)
        df = df.rolling(window=7).mean()
        data = df.to_numpy(dtype=float)[150:]
        normalization_idx = [0, 1]
        return data, normalization_idx

    net_params = [
        ("hidden_size", [1, 2, 4, 8, 16, 32, 64, 128, 256]),
        ("num_layers", [1, 2]),
    ]


# 特徴量 移動平均
#   カウント
#   緊急事態宣言(経過日数)
