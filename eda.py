#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

df = pd.read_csv("./data/SIGNATE COVID-19 Case Dataset (Tokyo) 2021 - 罹患者_東京_2021.csv")

#%% [markdown]
# それぞれの列でvalue_counts

#%%
for c in df:
    print(df[c].value_counts())
    print("-"*20)

#%% [markdown]
# それぞれの年代で日付別にvalue_counts
# 20代が一番多い

#%%
age_list = [i for i in np.unique(df["年代"])]
fig_ax_list = [plt.subplots() for _ in range(len(age_list) + 2)]
result_df = pd.DataFrame(columns=age_list)
for date, tmp_df in df.groupby("公表日"):
    count = tmp_df.groupby("年代")["年代"].count()
    count.name = date
    result_df = result_df.append(count)

# 年代別に1つずつプロット
for i, j in enumerate(result_df):
    fig = fig_ax_list[i][0]
    ax = fig_ax_list[i][1]
    ax.plot(result_df[j])
    ax.set_title(j)
    ax.set_ylim(0, 750)
    fig.canvas.draw()
    xlabel = ax.get_xticklabels()
    num = len(xlabel) // 7
    ax.set_xticks(np.arange(0, len(xlabel), num))
    ax.set_xticklabels(xlabel[::num], rotation=50)

# すべての年代をまとめてプロット
for i in result_df:
    fig = fig_ax_list[-2][0]
    ax = fig_ax_list[-2][1]
    ax.plot(result_df[i], label=i)
ax.set_title("all")
fig.canvas.draw()
ax.legend()
xlabel = ax.get_xticklabels()
num = len(xlabel) // 7
ax.set_xticks(np.arange(0, len(xlabel), num))
ax.set_xticklabels(xlabel[::num], rotation=50)

# 7日で移動平均
result_df_rolling = result_df.rolling(7).mean()
for i in result_df_rolling:
    fig = fig_ax_list[-1][0]
    ax = fig_ax_list[-1][1]
    ax.plot(result_df_rolling[i], label=i)
ax.set_title("all - rolling")
fig.canvas.draw()
ax.legend()
xlabel = ax.get_xticklabels()
num = len(xlabel) // 7
ax.set_xticks(np.arange(0, len(xlabel), num))
ax.set_xticklabels(xlabel[::num], rotation=50)

# stackplot
fig, ax = plt.subplots()
ax.stackplot(result_df_rolling.index, [i for i in result_df_rolling.values.T], labels=age_list, linewidth=0)
fig.canvas.draw()
ax.legend()
xlabel = ax.get_xticklabels()
num = len(xlabel) // 7
ax.set_xticks(np.arange(0, len(xlabel), num))
ax.set_xticklabels(xlabel[::num], rotation=50)
ax.set_title("all - stack")

#%% [markdown]
# 曜日別で公表日をカウント 月曜日が一番少ない

#%%
plt.figure()
df["公表日"] = pd.to_datetime(df["公表日"])
df["曜日"] = df["公表日"].dt.day_name()
result = df.groupby("曜日")["曜日"].count()
xlabel = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
plt.bar(xlabel, result[xlabel])
plt.xticks(rotation=30)
plt.title("曜日別")

#%% [markdown]
# 男女別にカウント
# 30代から50代まで男女の人数差が大きい → 外で働いている人が多いため？ → 就業者数のデータを用いれば相関がありそう

#%%
plt.figure()
sns.countplot(x="性別", data=df)
plt.figure()
sns.countplot(x="年代", data=df, hue="性別", order=age_list)
plt.xticks(rotation=30)

#%% [markdown]
# # その他
# - 天候が悪いと外に出ない → 感染者が少なくなる → 公表日と過去の天候を組み合わせてどれぐらいのラグがあるか発見できそう？
# - 変異株がいつ頃から流行りだしたか分析したい(方法は不明)
# - 緊急事態宣言の効果について