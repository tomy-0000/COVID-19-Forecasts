#%%
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

url1 ="https://docs.google.com/spreadsheets/d/1Ot0T8_YZ2Q0dORnKEhcUmuYCqZ1y81PIsIAMB7WZE8g/gviz/tq?tqx=out:csv&sheet=%E7%BD%B9%E6%82%A3%E8%80%85_%E6%9D%B1%E4%BA%AC_2020"
url2 = "https://docs.google.com/spreadsheets/d/1V1eJM1mupE9gJ6_k0q_77nlFoRuwDuBliMLcMdDMC_E/gviz/tq?tqx=out:csv&sheet=%E7%BD%B9%E6%82%A3%E8%80%85_%E6%9D%B1%E4%BA%AC_2021"

df = pd.concat([pd.read_csv(url1), pd.read_csv(url2)])
df = df[["公表日", "年代", "性別"]]
df = df[df["公表日"].notna()]
start = df["公表日"].iat[0]
end = df["公表日"].iat[-1]
index = pd.date_range(start=start, end=end)
count_series = pd.Series(0, index=index)
for date, tmp_df in df.groupby("公表日"):
    count_series[date] += len(tmp_df)
data = count_series.to_numpy()

DEVICE = torch.device("CUDA:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, num_layers=2, dropout=0.2, batch_first=True)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, 0, :])
        return y

class Sin(torch.utils.data.Dataset):
    def __init__(self, data, seq):
        self.seq = seq
        self.data = torch.from_numpy(data.reshape(-1, 1)).float()

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq], self.data[idx + self.seq]

    def __len__(self):
        return len(self.data) - self.seq

class Count(torch.utils.data.Dataset):
    def __init__(self, data, seq):
        self.seq = seq
        self.data = torch.from_numpy(data.reshape(-1, 1)).float()

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq], self.data[idx + self.seq]

    def __len__(self):
        return len(self.data) - self.seq

seq = 7
train_dataset = Count(data[:-30], seq)
val_dataset = Count(data[-30:], seq)
# import numpy as np
# x = np.arange(0, 13, 0.01)
# y = np.sin(x)
# train_dataset = Sin(y[:650], seq)
# val_dataset = Sin(y[650:], seq)
batch_size = 32
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

net = Net().to(DEVICE)
optimizer = torch.optim.Adam(net.parameters())
criterion = nn.MSELoss()
criterion2 = nn.L1Loss(reduction="sum")

result_dict = {"train_loss": [], "train_mae": [],
               "val_loss": [], "val_mae": []}
dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

epoch = 100

for i in tqdm(range(epoch)):
    for phase in ["train", "val"]:
        if phase == "train":
            net.train()
        else:
            net.eval()
        epoch_loss = 0
        epoch_mae = 0
        with torch.set_grad_enabled(phase == "train"):
            for inputs, labels in dataloader_dict[phase]:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item()*inputs.size(0)
                epoch_mae += criterion2(outputs, labels).detach()
            epoch_loss /= len(dataloader_dict[phase].dataset)
            epoch_mae /= len(dataloader_dict[phase].dataset)
            if i % 100 == 0:
                tqdm.write(f"{phase}_epoch_loss: {epoch_loss}")
                tqdm.write(f"{phase}_epoch_mae: {epoch_mae}")
            result_dict[phase+"_loss"].append(epoch_loss)
            result_dict[phase+"_mae"].append(epoch_mae)
plt.figure()
plt.plot(result_dict["train_loss"])
plt.plot(result_dict["val_loss"])
plt.figure()
plt.plot(result_dict["train_mae"])
plt.plot(result_dict["val_mae"])

for phase in ["train", "val"]:
    pred = []
    label = []
    with torch.set_grad_enabled(False):
        for inputs, labels in dataloader_dict[phase]:
            pred += net(inputs).detach().numpy().tolist()
            label += labels.numpy().tolist()
    plt.figure()
    plt.plot(pred)
    plt.plot(label)
