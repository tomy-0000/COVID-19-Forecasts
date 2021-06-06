import pandas as pd
import torch.nn as nn

def get_data(df):
    df = df.copy()
    df["day_name"] = df.index.day_name()
    df = pd.get_dummies(df, prefix="", prefix_sep="")
    columns = ["count", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df = df.loc[:, columns]
    data = df.to_numpy(dtype=float)
    return data

class Net2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y
