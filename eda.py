#%%
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

url1 = "https://docs.google.com/spreadsheets/d/1Ot0T8_YZ2Q0dORnKEhcUmuYCqZ1y81PIsIAMB7WZE8g/gviz/tq?tqx=out:csv&sheet=%E7%BD%B9%E6%82%A3%E8%80%85_%E6%9D%B1%E4%BA%AC_2020"
url2 = "https://docs.google.com/spreadsheets/d/1V1eJM1mupE9gJ6_k0q_77nlFoRuwDuBliMLcMdDMC_E/gviz/tq?tqx=out:csv&sheet=%E7%BD%B9%E6%82%A3%E8%80%85_%E6%9D%B1%E4%BA%AC_2021"

df = pd.concat([pd.read_csv(url1), pd.read_csv(url2)])
print(df.columns)
display(df.head())

#%% [markdown]
# # 前処理
# - 最終的に公表日・年代・性別のみとなる

#%%
df = df[["公表日", "年代", "性別"]]
df = df.dropna()
df = df[~df["年代"].isin(["非公開", "非公表"])]
df = df[df["性別"] != "非公表"]
for i in df:
    display(df[i].value_counts())

#%% [markdown]
# # 公表日別にカウント
# - 20代が一番多い
# - 波がある

#%%
# result_df・result_df_rolllingの作成
age_list = [i for i in np.unique(df["年代"])]
start = df["公表日"].iat[0]
end = df["公表日"].iat[-1]
index = pd.date_range(start=start, end=end)
count_age_df = pd.DataFrame(0, columns=age_list, index=index)
for date, tmp_df in df.groupby("公表日"):
    date = pd.to_datetime(date)
    count_age_df.loc[date, :] += tmp_df.groupby("年代")["年代"].count()
count_age_df = count_age_df.fillna(0)
count_age_rolling_df = count_age_df.rolling(7).mean().fillna(0)
count_age_df = count_age_df.astype(int)
count_age_rolling_df = count_age_rolling_df
count_df = count_age_df.sum(axis=1)
count_rolling_df = count_df.rolling(7).mean().fillna(0)

#%%
# 年代別・公表日別にカウント
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(10)
for i in count_age_df:
    ax.plot(count_age_df[i], label=i)
ax.set_title("公表日別カウント")
ax.legend()

# 年代別・公表日別にカウント(7日間移動平均)
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(10)
for i in count_age_rolling_df:
    ax.plot(count_age_rolling_df[i], label=i)
ax.set_title("公表日別カウント(7日間移動平均)")
ax.legend()

# 年代別・公表日別にカウント(7日間移動平均・stackplot)
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(10)
ax.stackplot(
    count_age_rolling_df.index,
    [i for i in count_age_rolling_df.values.T],
    labels=age_list,
    linewidth=0,
)
ax.legend()

# 年代別にカウント
fig, ax = plt.subplots()
sns.countplot(x="年代", data=df, color="C0", order=age_list)
plt.xticks(rotation=30)
plt.ylabel("count(人)")
plt.title("年代別カウント")

#%% [markdown]
# # 曜日別にカウント
# - 月曜日が一番少ない
#   - 「前日が日曜日で休みの医療機関が多く、持ち込まれる検体数が少ないため、月曜日は確認される人数が少ない傾向にあります。」(https://www.news24.jp/articles/2020/04/14/07625302.html - 日テレNEWS24)
# - 曜日による男女の違いはなさそう
# - 曜日による年代の違いはなさそう

#%%
# 曜日列の作成
df["公表日"] = pd.to_datetime(df["公表日"])
df["曜日"] = df["公表日"].dt.day_name()
xlabel = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# 曜日別カウント
plt.figure()
sns.countplot(x="曜日", data=df, order=xlabel, color="C0")
plt.xticks(rotation=30)
plt.title("曜日別カウント")
plt.ylabel("count(人)")

# 曜日別カウント(男女別)
plt.figure()
sns.countplot(x="曜日", data=df, hue="性別", order=xlabel)
plt.xticks(rotation=30)
plt.title("曜日別カウント(男女別)")
plt.ylabel("count(人)")

plt.figure(figsize=(14, 10))
sns.countplot(x="曜日", data=df, hue="年代", order=xlabel, hue_order=age_list)
plt.title("曜日別カウント(年代別)")
plt.ylabel("count(人)")

#%% [markdown]
# # 男女別にカウント
# - 30代から50代まで男女の人数差が大きい → 外で働いている人が多いため？ → 就業者数のデータを用いれば相関がありそう

#%%
plt.figure()
sns.countplot(x="性別", data=df)
plt.ylabel("count(人)")
plt.title("男女別カウント")
plt.figure()
sns.countplot(x="年代", data=df, hue="性別", order=age_list)
plt.xticks(rotation=30)
plt.title("年代別カウント(男女別)")
plt.ylabel("count(人)")

#%%
plt.figure()
kinnkyu = [
    ["2020-04-07", "2020-05-25"],
    ["2021-01-08", "2021-03-21"],
    ["2021-04-25", "2021-05-31"],
]
sns.lineplot(data=count_rolling_df)
for x1, x2 in kinnkyu:
    x1 = pd.to_datetime(x1)
    x2 = pd.to_datetime(x2)
    x = pd.date_range(start=x1, end=x2)
    plt.fill_between(x, 0, count_rolling_df.max(), alpha=0.3, color="C1")
plt.xticks(rotation=45)
plt.ylabel("count(人)")
plt.title("公表日別カウント(移動平均)")

#%% [markdown]
# # その他
# - 天候が悪いと外に出ない → 感染者が少なくなる → 公表日と過去の天候を組み合わせてどれぐらいのラグがあるか発見できそう？
# - 変異株がいつ頃から流行りだしたか分析したい(方法は不明)
# - 緊急事態宣言の効果について
