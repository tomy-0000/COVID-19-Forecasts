import pandas as pd
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, hidden_size, num_layers, predict_seq):
        super().__init__()
        self.lstm = nn.LSTM(8, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, predict_seq)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y

    @staticmethod
    def get_data(i, use_seq, predict_seq):
        df = pd.read_csv("data_use/count.csv", parse_dates=True, index_col=0)[["東京都"]]
        df["day_name"] = df.index.day_name()
        df = pd.get_dummies(df, prefix="", prefix_sep="")
        columns = ["東京都", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df = df.loc[:, columns]
        data = df.values.astype(float)
        total_seq = use_seq + predict_seq
        train_data = data[:-2*total_seq]
        val_data = data[-2*total_seq:-total_seq]
        test_data = data[-total_seq:]
        return [train_data, val_data, test_data]

    normalization_idx = [0]
    net_params = {
        "hidden_size": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        "num_layers": [1, 2]
    }

# 特徴量
#   カウント
#   曜日(one hot encoding)
