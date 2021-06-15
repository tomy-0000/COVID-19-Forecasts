import pandas as pd
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y

    @staticmethod
    def get_data():
        df = pd.read_csv("https://raw.githubusercontent.com/tomy-0000/COVID-19-Forecasts/master/data/count.csv", parse_dates=True, index_col=0)
        df = df.rolling(window=14).mean()
        data = df.to_numpy(dtype=float)[150:]
        return data

    dataset_config = {"seq": 30,
                      "val_test_len": 30,
                      "batch_size": 10000,
                      "normalization_idx": [0]}

    net_config = {"hidden_size": 32,
                  "num_layers": 1}

# 特徴量 移動平均
#   カウント
