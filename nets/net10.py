import pandas as pd
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(10, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y

    @staticmethod
    def get_data():
        df1 = pd.read_csv("./data/count_tokyo.csv", parse_dates=True, index_col=0)
        df2 = pd.read_csv("./data/weather.csv", parse_dates=True, index_col=0)
        df3 = pd.read_csv("./data/emergency.csv", parse_dates=True, index_col=0)
        df = pd.concat([df1, df2, df3], axis=1)
        df = df.rolling(window=7).mean()
        data = df.to_numpy(dtype=float)[150:]
        normalization_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        return data, normalization_idx

    net_params = [
        ("hidden_size", [1, 2, 4, 8, 16, 32, 64, 128, 256]),
        ("num_layers", [1, 2])
    ]

# 特徴量 移動平均
#   カウント
#   気温
#   降水量
#   風速
#   現地気圧
#   相対温度
#   蒸気圧
#   天気
#   雲量
#   緊急事態宣言(経過日数)
