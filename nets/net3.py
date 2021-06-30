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
        df = pd.read_csv("./data/count_tokyo.csv", parse_dates=True, index_col=0)
        df["day_name"] = df.index.day_name()
        le = LabelEncoder()
        df["day_name"] = le.fit_transform(df.index.day_name())
        data = df.to_numpy(dtype=float)[150:]
        normalization_idx = [0]
        return data, normalization_idx

    net_params = [
        ("weather_embedding_dim", [1, 2, 4, 8, 16, 32, 64, 128, 256]),
        ("hidden_size", [1, 2, 4, 8, 16, 32, 64, 128, 256]),
        ("num_layers", [1, 2])
    ]

# 特徴量
#   カウント
#   曜日(Embedding)