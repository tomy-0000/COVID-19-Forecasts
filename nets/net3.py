import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, day_embedding_dim, hidden_size, num_layers, predict_seq):
        super().__init__()
        self.day_embedding = nn.Embedding(7, day_embedding_dim)
        feature_num = 1 + day_embedding_dim
        self.lstm = nn.LSTM(feature_num, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, predict_seq)

    def forward(self, x):
        x, day = x[:, :, [0]], x[:, :, 1].long()
        day = self.day_embedding(day)
        x = torch.cat([x, day], dim=2)
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y

    @staticmethod
    def get_data(i, use_seq, predict_seq):
        df = pd.read_csv("data_use/count.csv", parse_dates=True, index_col=0)[["東京都"]]
        df["day_name"] = df.index.day_name()
        le = LabelEncoder()
        df["day_name"] = le.fit_transform(df.index.day_name())
        data = df.values.astype(float)
        total_seq = use_seq + predict_seq
        train_data = data[: -2 * total_seq]
        val_data = data[-2 * total_seq : -total_seq]
        test_data = data[-total_seq:]
        return [train_data, val_data, test_data]

    normalization_idx = [0]
    net_params = {
        "day_embedding_dim": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        "hidden_size": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        "num_layers": [1, 2],
    }


# 特徴量
#   カウント
#   曜日(Embedding)
