#%%
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

url1 ="https://docs.google.com/spreadsheets/d/1Ot0T8_YZ2Q0dORnKEhcUmuYCqZ1y81PIsIAMB7WZE8g/gviz/tq?tqx=out:csv&sheet=%E7%BD%B9%E6%82%A3%E8%80%85_%E6%9D%B1%E4%BA%AC_2020"
url2 = "https://docs.google.com/spreadsheets/d/1V1eJM1mupE9gJ6_k0q_77nlFoRuwDuBliMLcMdDMC_E/gviz/tq?tqx=out:csv&sheet=%E7%BD%B9%E6%82%A3%E8%80%85_%E6%9D%B1%E4%BA%AC_2021"

df = pd.concat([pd.read_csv(url1), pd.read_csv(url2)])
df = df[["公表日", "年代", "性別"]]
df = df[df["公表日"].notna()]
start = df["公表日"].iat[0]
end = df["公表日"].iat[-1]
index = pd.date_range(start=start, end=end)
df2 = pd.DataFrame(0, columns=["count", "day_name"], index=index)
for date, tmp_df in df.groupby("公表日"):
    df2.loc[date, "count"] += len(tmp_df)
df2["day_name"] = df2.index.day_name()
df2 = pd.get_dummies(df2, prefix="", prefix_sep="")
data = df2.to_numpy()
# data_rolling = df.rolling(7).mean().dropna().to_numpy()

#%%
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

class Net(nn.Module):
    def __init__(self, feature_num):
        super().__init__()
        self.lstm = nn.LSTM(feature_num, 32, num_layers=1, batch_first=True)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, seq):
        self.seq = seq
        self.data = torch.from_numpy(data).float()

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq], self.data[idx + self.seq, 0].unsqueeze(0)

    def __len__(self):
        return len(self.data) - self.seq

    def get_label(self):
        return self.data[:, 0].numpy()

    def get_feature(self, idx):
        return self.data[idx + self.seq, 1:].numpy()

class TrainVal:
    def __init__(self, data, seq, val_len, batch_size=32, normalization_idx=[0]):
        self.data = data
        self.feature_num = data.shape[1]
        self.seq = seq
        self.val_len = val_len + seq
        train_data = data[:-val_len]
        val_data = data[-val_len:]
        self.normalization_idx = normalization_idx
        self.mean = np.mean(train_data[:, normalization_idx], axis=0)
        self.std = np.std(train_data[:, normalization_idx], axis=0)
        train_data[normalization_idx] = (train_data[normalization_idx] - self.mean)/self.std
        val_data[normalization_idx] = (val_data[normalization_idx] - self.mean)/self.std
        self.train_dataset = Dataset(train_data, seq)
        self.val_dataset = Dataset(val_data, seq)
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size)
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size)
        self.dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

def run(train_val, epoch):
    dataloader_dict = train_val.dataloader_dict

    net = Net(train_val.feature_num)
    net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters())
    criterion = nn.MSELoss()
    result_dict = {"train_loss": [], "train_mae": [],
                   "val_loss": [], "val_mae": []}
    best_dict = {"mae_loss": 1e10, "state_dict": None, "epoch": 0}

    show_progress = epoch // 100
    for i in tqdm(range(epoch)):
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            epoch_loss = 0
            epoch_mae = 0
            with torch.set_grad_enabled(phase == "train"):
                for inputs, label in dataloader_dict[phase]:
                    inputs = inputs.to(DEVICE)
                    label = label.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, label)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item()*inputs.size(0)
                    epoch_mae += F.l1_loss(outputs, label, reduction="sum").item()
            epoch_loss /= len(dataloader_dict[phase].dataset)
            epoch_mae /= len(dataloader_dict[phase].dataset)
            if i % show_progress == 0:
                tqdm.write(f"{i}_{phase}_epoch_loss: {epoch_loss}")
                tqdm.write(f"{i}_{phase}_epoch_mae: {epoch_mae}")
            result_dict[phase+"_loss"].append(epoch_loss)
            result_dict[phase+"_mae"].append(epoch_mae)
            if phase == "val" and epoch_mae < best_dict["mae_loss"]:
                best_dict["mae_loss"] = epoch_mae
                best_dict["state_dict"] = net.state_dict()
                best_dict["epoch"] = i + 1

    plt.figure(figsize=(12, 8))
    plt.plot(result_dict["train_loss"])
    plt.plot(result_dict["val_loss"])
    plt.title("mse")
    plt.figure(figsize=(12, 8))
    plt.plot(result_dict["train_mae"])
    plt.plot(result_dict["val_mae"])
    plt.title("mae")

    print("best_epoch:", best_dict["epoch"])
    # net.load_state_dict(best_dict["state_dict"])
    net.to("cpu")
    net.eval()
    seq = train_val.seq

    train_dataset = train_val.train_dataset
    pred_list = train_dataset[0][0].numpy()
    label_list = train_dataset.get_label()
    with torch.set_grad_enabled(False):
        for i in range(len(train_dataset)):
            inputs = torch.from_numpy(pred_list[-seq:]).unsqueeze(0)
            out = net(inputs)
            out = out.numpy()
            feature = train_dataset.get_feature(i)
            out = np.append(out, feature)
            pred_list = np.vstack([pred_list, out])
    pred_list = pred_list[:, 0]
    plt.figure(figsize=(12, 8))
    plt.plot(seq, pred_list[seq], ".", c="C0", markersize=20)
    plt.plot(pred_list, label="predict")
    plt.plot(label_list, label="gt")
    plt.title("pred_train")
    plt.legend()

    val_dataset = train_val.val_dataset
    pred_list = val_dataset[0][0].numpy()
    label_list = val_dataset.get_label()
    with torch.set_grad_enabled(False):
        for i in range(len(val_dataset)):
            inputs = torch.from_numpy(pred_list[-seq:]).unsqueeze(0)
            out = net(inputs)
            out = out.numpy()
            feature = val_dataset.get_feature(i)
            out = np.append(out, feature)
            pred_list = np.vstack([pred_list, out])
    pred_list = pred_list[:, 0]
    plt.figure(figsize=(12, 8))
    plt.plot(seq, pred_list[seq], ".", c="C0", markersize=20)
    plt.plot(pred_list, label="predict")
    plt.plot(label_list, label="gt")
    plt.title("pred_train")
    plt.legend()

seq = 5
val_len = 30
batch_size = 200

train_val = TrainVal(data[100:], seq, val_len, batch_size)
epoch = 100
run(train_val, epoch)

#%%
# seq = 30
# val_len = 30

# val_len += seq
# train_dataset = Count(data_rolling[:-val_len], seq)
# val_dataset = Count(data_rolling[-val_len:], seq)
# batch_size = 8192
# epoch = 50000
# run(train_dataset, val_dataset, batch_size, epoch, seq)

np.mean(data[:, [0, 1]], axis=0)