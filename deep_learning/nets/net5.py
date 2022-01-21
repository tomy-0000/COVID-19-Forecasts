import pandas as pd
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, hidden_size, num_layers, predict_seq):
        super().__init__()
        self.lstm = nn.LSTM(16, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, predict_seq)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y

    @staticmethod
    def get_data():
        df1 = pd.read_csv("./data/count_tokyo.csv", parse_dates=True, index_col=0)
        df1["day_name"] = df1.index.day_name()
        df1 = pd.get_dummies(df1, prefix="", prefix_sep="")
        columns = [
            "count",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        df1 = df1.loc[:, columns]
        df2 = pd.read_csv("./data/weather.csv", parse_dates=True, index_col=0)
        df = pd.concat([df1, df2], axis=1)
        data = df.to_numpy(dtype=float)[150:]
        normalization_idx = [0, 8, 9, 10, 11, 12, 13, 14, 15]
        return data, normalization_idx

    net_params = [
        ("hidden_size", [1, 2, 4, 8, 16, 32, 64, 128, 256]),
        ("num_layers", [1, 2]),
    ]


# 特徴量
#   カウント
#   曜日(Embedding)
#   気温
#   降水量
#   風速
#   現地気圧
#   相対温度
#   蒸気圧
#   天気
#   雲量
