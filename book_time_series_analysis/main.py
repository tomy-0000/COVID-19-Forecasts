#%%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa import ar_model, stattools

warnings.simplefilter("ignore")
sns.set()
import japanize_matplotlib

df = pd.read_csv("data/use/count.csv", index_col=0)["Tokyo"]
display(df)
df_icecream = pd.read_csv("data/use/icecream.csv", index_col=0)["count"]
display(df_icecream)
df_icecream_diff = df_icecream.diff().dropna()

plt.plot(df)
plt.xticks("")
#%% [markdown]
# # 1.8 移動平均乖離率

#%%
df_ma = df.rolling(14).mean()
df_ma_diff = ((df - df_ma) / df_ma * 100).dropna()
# df_ma_diff = df_ma_diff.replace(np.inf, 1000).replace(-np.inf, -1000)
plt.plot(df_ma_diff)
plt.xticks("")

plt.figure()
plt.plot(df, label="オリジナル", alpha=0.7)
plt.plot(df_ma, label="移動平均", alpha=0.7)
plt.xticks("")
plt.legend()

#%% [markdown]
# # 自己相関・偏自己相関

#%%
sm.graphics.tsa.plot_acf(df)
sm.graphics.tsa.plot_pacf(df)
""

#%% [markdown]
# # 2.1.1 最小二乗法

#%%
x = np.arange(len(df))
a = sum((df - df.mean()) * (x - x.mean())) / sum((x - x.mean()) ** 2)
y = a * x
plt.plot(x, y)
plt.plot(df)
plt.xticks("")

#%% [markdown]
# # 2.2 ARモデル(statsmodel)

#%%
for prefix, df_tmp in zip(["original_", "diff_"], [df, df.diff().dropna()]):
    ctt = stattools.adfuller(df_tmp, regression="ctt")
    ct = stattools.adfuller(df_tmp, regression="ct")
    c = stattools.adfuller(df_tmp, regression="c")
    nc = stattools.adfuller(df_tmp, regression="nc")
    print(prefix + "ctt:")
    print(ctt[1])
    print(prefix + "ct:")
    print(ct[1])
    print(prefix + "c:")
    print(c[1])
    print(prefix + "nc:")
    print(nc[1])
    print()


min_lag = 0
min_aic = 1000
for i in range(1, 21):
    results = ar_model.AutoReg(df, i).fit()
    print("lag =", i, ", aic =", results.aic)
    if results.aic < min_aic:
        min_lag = i
        min_aic = results.aic
print("min_lag =", min_lag, "min_aic =", min_aic)
results = ar_model.AutoReg(df[:-30].values, min_lag).fit()
predict = results.predict(results.params, len(df) - 30 - min_lag, len(df), dynamic=True)

plt.figure(figsize=(20, 10), dpi=300)
plt.plot(df)
plt.plot(min_lag + np.arange(len(results.fittedvalues)), results.fittedvalues)
plt.plot(min_lag + len(results.fittedvalues) + np.arange(len(predict)), predict)
plt.xticks("")

#%% [markdown]
# 100日間のデータに対して1日〜70日を訓練データ、71日〜100日をテストデータとする predictメソッドに(71 - min_lag)を渡す

#%% [markdown]
# # 2.2 ARモデル(フルスクラッチ)


#%%
class AR:
    def __init__(self, df, p, title, train_size=80):
        self.p = p
        self.title = title
        df = pd.DataFrame(df.copy())
        for i in range(1, p + 1):
            df[i] = df["count"].shift(i)
        df = df.dropna(how="any").astype(float)
        df = df[df.sum(axis=1) != 0]
        t = df.iloc[:, 0].copy()
        X = df.values.copy()
        X[:, 0] = 1.0
        self.train_X, self.test_X = X[:train_size], X[train_size:]
        self.train_t, self.test_t = t[:train_size], t[train_size:]

    def fit(self):
        train_X = self.train_X
        train_t = self.train_t
        a = np.linalg.inv(train_X.T @ train_X) @ train_X.T @ train_t
        self.a = a

    def predict(self, x):
        print(x)
        print(self.a)
        print(x @ self.a)
        return x @ self.a

    def predict_repeat(self, t, l):
        y = [1] + t[-self.p :].tolist()
        for _ in range(l):
            y.append(self.predict([1] + y[-self.p :]))
        return y[self.p :]

    def plot_predict(self):
        plt.figure(dpi=300)
        train_index = np.arange(len(self.train_X))
        test_index = np.arange(len(self.test_X) + 1) + len(train_index)
        all_index = np.arange(len(self.train_X) + len(self.test_X))
        plt.plot(all_index, np.concatenate([self.train_t, self.test_t]), c="C0", label="gt")
        plt.plot(
            test_index,
            self.predict_repeat(self.train_t, len(self.test_X)),
            c="C1",
            linestyle="--",
            label="test_y",
        )
        plt.legend()
        plt.title(self.title)


ice_ar = AR(df_icecream, 11, "ice")
ice_ar.fit()
ice_ar.plot_predict()

ice_diff_ar = AR(df_icecream_diff, 11, "ice_diff")
ice_diff_ar.fit()
ice_diff_ar.plot_predict()

covid19_ar = AR(df, 14, "covid19", train_size=365)
covid19_ar.fit()
covid19_ar.plot_predict()
