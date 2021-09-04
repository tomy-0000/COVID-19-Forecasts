import pandas as pd
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, hidden_size, num_layers, predict_seq):
        super().__init__()
        self.lstm = nn.LSTM(9, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, predict_seq)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y

    @staticmethod
    def get_data(i, use_seq, predict_seq):
        df1 = pd.read_csv("data_use/count.csv", parse_dates=True, index_col=0)[["東京都"]]
        df2 = pd.read_csv("data_use/weather.csv", parse_dates=True, index_col=0)
        df = pd.concat([df1, df2], axis=1)
        data = df.values.astype(float)
        total_seq = use_seq + predict_seq
        train_data = data[:-2*total_seq]
        val_data = data[-2*total_seq:-total_seq]
        test_data = data[-total_seq:]
        return [train_data, val_data, test_data]

    normalization_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    net_params = {
        "hidden_size": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        "num_layers": [1, 2]
    }

# 特徴量
#   カウント
#   気温
#   降水量
#   風速
#   現地気圧
#   相対温度
#   蒸気圧
#   天気
#   雲量
