import pandas as pd
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(16, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y

    @staticmethod
    def get_data():
        df1 = pd.read_csv("https://raw.githubusercontent.com/tomy-0000/COVID-19-Forecasts/master/data/count.csv", parse_dates=True, index_col=0)
        df1["day_name"] = df1.index.day_name()
        df1 = pd.get_dummies(df1, prefix="", prefix_sep="")
        columns = ["count", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df1 = df1.loc[:, columns]
        df2 = pd.read_csv("https://raw.githubusercontent.com/tomy-0000/COVID-19-Forecasts/master/data/weather.csv", parse_dates=True, index_col=0)
        df = pd.concat([df1, df2], axis=1)
        data = df.to_numpy(dtype=float)[150:]
        return data

    dataset_config = {"seq": 30,
                      "val_test_len": 30,
                      "batch_size": 10000,
                      "normalization_idx": [0, 8, 9, 10, 11, 12, 13, 14, 15]}

    net_config = {"hidden_size": 32,
                  "num_layers": 1}

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
