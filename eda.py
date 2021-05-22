#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

df = pd.concat([pd.read_csv("data/tokyo_2020.csv"), pd.read_csv("data/tokyo_2021.csv")])
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

#%%
for i in df:
    display(df[i].value_counts())

#%% [markdown]
# # 公表日別にカウント
# - 20代が一番多い
# - 波がある

#%%
# result_df・result_df_rolllingの作成
age_list = [i for i in np.unique(df["年代"])]
result_df = pd.DataFrame(columns=age_list)
for date, tmp_df in df.groupby("公表日"):
    tmp_series = pd.Series(0, index=age_list)
    count = tmp_series + tmp_df.groupby("年代")["年代"].count()
    count = count.fillna(0)
    count.name = date
    result_df = result_df.append(count)
result_df_rolling = result_df.rolling(7).mean()

# 年代別・公表日別にカウント
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(10)
for i in result_df:
    ax.plot(result_df[i], label=i)
ax.set_title("公表日別カウント")
fig.canvas.draw()
ax.legend()
xlabel = ax.get_xticklabels()
num = len(xlabel) // 7
ax.set_xticks(np.arange(0, len(xlabel), num))
ax.set_xlabel("日付")
ax.set_ylabel("count(人)")
ax.set_xticklabels(xlabel[::num], rotation=50)

# 年代別・公表日別にカウント(7日間移動平均)
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(10)
for i in result_df_rolling:
    ax.plot(result_df_rolling[i], label=i)
ax.set_title("公表日別カウント(7日間移動平均)")
fig.canvas.draw()
ax.legend()
xlabel = ax.get_xticklabels()
num = len(xlabel) // 7
ax.set_xticks(np.arange(0, len(xlabel), num))
ax.set_xlabel("日付")
ax.set_ylabel("count(人)")
ax.set_xticklabels(xlabel[::num], rotation=50)

# 年代別・公表日別にカウント(7日間移動平均・stackplot)
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(10)
ax.stackplot(result_df_rolling.index, [i for i in result_df_rolling.values.T], labels=age_list, linewidth=0)
fig.canvas.draw()
ax.legend()
xlabel = ax.get_xticklabels()
num = len(xlabel) // 7
ax.set_xticks(np.arange(0, len(xlabel), num))
ax.set_xticklabels(xlabel[::num], rotation=50)
ax.set_xlabel("日付")
ax.set_ylabel("count(人)")
ax.set_title("公表日別カウント(7日間移動平均・積み上げ)")

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

#%% [markdown]
# # その他
# - 天候が悪いと外に出ない → 感染者が少なくなる → 公表日と過去の天候を組み合わせてどれぐらいのラグがあるか発見できそう？
# - 変異株がいつ頃から流行りだしたか分析したい(方法は不明)
# - 緊急事態宣言の効果について