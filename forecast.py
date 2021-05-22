#%%
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

df = pd.concat([pd.read_csv("data/tokyo_2020.csv"), pd.read_csv("data/tokyo_2021.csv")])
df = df[["公表日", "年代", "性別"]]
df = df[df["公表日"].notna()]
start = df["公表日"].iat[0]
end = df["公表日"].iat[-1]
index = pd.date_range(start=start, end=end)
count_series = pd.Series(0, index=index)
for date, tmp_df in df.groupby("公表日"):
    count_series[date] += len(tmp_df)
count = count_series.to_numpy()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, num_layers=2, dropout=0.2, batch_first=True)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, 0, :])
        return y

class Count(torch.utils.data.Dataset):
    def __init__(self, count, seq):
        self.seq = seq
        self.data = torch.from_numpy(count.reshape(-1, 1)).float()

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq], self.data[idx + self.seq]

    def __len__(self):
        return len(self.data) - self.seq

seq = 7
train_dataset = Count(count[:-30], seq)
val_dataset = Count(count[30:], seq)
batch_size = 32
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

net = Net()
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
            if i % epoch//10 == 0:
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
