import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, weather_embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.weather_embedding = nn.Embedding(7, weather_embedding_dim)
        feature_num = 1 + weather_embedding_dim
        self.lstm = nn.LSTM(feature_num, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, weather = x[:, :, [0]], x[:, 0:, 1].long()
        weather = self.weather_embedding(weather)
        x = torch.cat([x, weather], dim=2)
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y

    @staticmethod
    def get_data():
        df = pd.read_csv("https://raw.githubusercontent.com/tomy-0000/COVID-19-Forecasts/master/data/count.csv", parse_dates=True, index_col=0)
        df["day_name"] = df.index.day_name()
        le = LabelEncoder()
        df["day_name"] = le.fit_transform(df.index.day_name())
        data = df.to_numpy(dtype=float)[150:]
        return data

    dataset_config = {"seq": 30,
                      "val_test_len": 30,
                      "batch_size": 10000,
                      "normalization_idx": [0]}

    net_config = {"weather_embedding_dim": 8,
                  "hidden_size": 32,
                  "num_layers": 1}

# 特徴量
#   カウント
#   曜日(Embedding)