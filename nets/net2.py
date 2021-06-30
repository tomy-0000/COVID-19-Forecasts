import pandas as pd
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(8, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y

    @staticmethod
    def get_data():
        df = pd.read_csv("./data/count_tokyo.csv", parse_dates=True, index_col=0)
        df["day_name"] = df.index.day_name()
        df = pd.get_dummies(df, prefix="", prefix_sep="")
        columns = ["count", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df = df.loc[:, columns]
        data = df.to_numpy(dtype=float)[150:]
        normalization_idx = [0]
        return data, normalization_idx

    net_params = [
        ("hidden_size", [1, 2, 4, 8, 16, 32, 64, 128, 256]),
        ("num_layers", [1, 2])
    ]

# 特徴量
#   カウント
#   曜日(one hot encoding)
