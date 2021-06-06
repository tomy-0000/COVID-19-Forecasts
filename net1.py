#%%
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn as nn
from utils import TrainValTest, run
sns.set(font="IPAexGothic")

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
columns = ["count", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df2 = df2.loc[:, columns]
data = df2.to_numpy(dtype=float)

#%%
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.linear(x[:, -1, :])
        return y

mae_list = []
epoch = 30000
repeat_num = 2

seq = 14
val_test_len = 30
batch_size = 200

# 曜日無し
train_val_test = TrainValTest(data[200:, [0]], seq, val_test_len, batch_size)
tmp = []
net_args = {"input_size": 1,
            "hidden_size": 32,
            "num_layers": 1}
run(Net, net_args, train_val_test, epoch, use_best=True, plot=True, log=False, patience=500)
for _ in tqdm(range(repeat_num)):
    tmp.append(run(Net, net_args, train_val_test, epoch, use_best=True, plot=False, log=False, patience=500))
mae_list.append(tmp)

# 曜日有り
train_val_test = TrainValTest(data[200:], seq, val_test_len, batch_size)
tmp = []
net_args = {"input_size": 8,
            "hidden_size": 32,
            "num_layers": 1}
run(Net, net_args, train_val_test, epoch, use_best=True, plot=True, log=False, patience=500)
for _ in tqdm(range(repeat_num)):
    tmp.append(run(Net, net_args, train_val_test, epoch, use_best=True, plot=False, log=False, patience=500))
mae_list.append(tmp)

columns = ["曜日なし", "曜日あり"]
result_df = pd.DataFrame({i: j for i, j in zip(columns, mae_list)})

plt.figure()
sns.boxplot(data=result_df)
sns.swarmplot(data=result_df, color="white", size=7, edgecolor="black", linewidth=2)