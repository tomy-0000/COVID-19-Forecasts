#%%
# https://www.niid.go.jp/niid/ja/diseases/ka/corona-virus/2019-ncov/2502-idsc/iasr-in/10465-496d04.html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data/use/count.csv", index_col=0, parse_dates=True)

df = df.rolling(7).mean()
for i in range(3, 8):
    df[f"Rt_{i}"] = df["count"] / df["count"].shift(i)
df = df.dropna(how="any")

df = df[~(df == np.inf).sum(axis=1).astype(bool)]

for i in range(3, 8):
    plt.plot(df[f"Rt_{i}"].iloc[-260:-170], alpha=0.5, label=f"Rt_{i}")
    # plt.plot(df[f"Rt_{i}"], alpha=0.5, label=f"Rt_{i}")
plt.legend()
plt.xticks(rotation=45)
plt.ylim(0)
plt.axhline(1.00, c="black", alpha=0.5)
