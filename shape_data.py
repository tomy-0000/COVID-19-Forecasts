#%%
import datetime
import glob
import pandas as pd

#%%
url = f"https://docs.google.com/spreadsheets/d/10MFfRQTblbOpuvOs_yjIYgntpMGBg592dL8veXoPpp4/gviz/tq?tqx=out:csv&sheet=%E7%BD%B9%E6%82%A3%E8%80%85%E7%B5%B1%E8%A8%88"
df = pd.read_csv(url)
df = df.set_index("日付", drop=True)

for i in df:
    tmp = df[i].astype(str)
    tmp = tmp.str.replace(",", "")
    df[i] = pd.to_numeric(tmp, errors="coerce")
df.to_csv("./data_raw/count.csv")

#%%
url1 ="https://docs.google.com/spreadsheets/d/1Ot0T8_YZ2Q0dORnKEhcUmuYCqZ1y81PIsIAMB7WZE8g/gviz/tq?tqx=out:csv&sheet=%E7%BD%B9%E6%82%A3%E8%80%85_%E6%9D%B1%E4%BA%AC_2020"
url2 = "https://docs.google.com/spreadsheets/d/1V1eJM1mupE9gJ6_k0q_77nlFoRuwDuBliMLcMdDMC_E/gviz/tq?tqx=out:csv&sheet=%E7%BD%B9%E6%82%A3%E8%80%85_%E6%9D%B1%E4%BA%AC_2021"

df = pd.concat([pd.read_csv(url1), pd.read_csv(url2)])
df = df[["公表日", "年代", "性別"]]
df = df[df["公表日"].notna()]
start = df["公表日"].iat[0]
end = df["公表日"].iat[-1]
index = pd.date_range(start=start, end=end)
df2 = pd.DataFrame(0, columns=["count"], index=index)
df2.index.name = "date"
for date, tmp_df in df.groupby("公表日"):
    df2.loc[date, "count"] += len(tmp_df)
df2 = df2[datetime.datetime(2020, 3, 1) <= df2.index <= datetime.datetime(2021, 5, 31)]
df2.to_csv("./data/count_tokyo.csv")

#%%
csv_list = sorted(glob.glob("./gitignore/202*.csv"))
df = pd.DataFrame()
usecols = [0, 1, 4, 8, 13, 16, 19, 22, 25]
names = ["date", "気温", "降水量", "風速", "現地気圧", "相対湿度", "蒸気圧", "天気", "雲量"]
for csv in csv_list:
    tmp_df = pd.read_csv(csv, encoding="shift-jis", skiprows=3, usecols=usecols, index_col=0, names=names).iloc[3:, :]
    tmp_df.index = pd.to_datetime(tmp_df.index)
    use = (7 <= tmp_df.index.hour) & (tmp_df.index.hour <= 15)
    tmp_df = tmp_df[use].fillna(method="bfill")
    df = pd.concat([df, tmp_df])

df["雲量"] = df["雲量"].str.extract("(\d+)")
dtypes = [float, float, float, float, int, float, int, int]
df = df.astype({i: j for i, j in zip(names[1:], dtypes)})
sorted(df["天気"].unique())
df["天気"] = df["天気"].map({j: i for i, j in enumerate(sorted(df["天気"].unique()))})
df = df.resample("D").mean()
df = df[df.index >= datetime.datetime(2020, 3, 31)]
df.to_csv("./data/weather.csv")

#%%
index = pd.date_range(start=datetime.datetime(2020, 3, 31), end=datetime.datetime(2021, 5, 31))
df = pd.DataFrame(0, columns=["緊急事態宣言"], index=index)
start1 = datetime.datetime(2020, 4, 7)
end1 = datetime.datetime(2020, 5, 25)
df.loc[start1:end1, "緊急事態宣言"] = list(range(1, (end1 - start1).days + 2))

start1 = datetime.datetime(2020, 4, 7)
end1 = datetime.datetime(2020, 5, 25)
df.loc[start1:end1, "緊急事態宣言"] = list(range(1, (end1 - start1).days + 2))

start2 = datetime.datetime(2021, 1, 8)
end2 = datetime.datetime(2021, 3, 21)
df.loc[start2:end2, "緊急事態宣言"] = list(range(1, (end2 - start2).days + 2))

start3 = datetime.datetime(2021, 4, 25)
end3 = datetime.datetime(2021, 5, 31)
df.loc[start3:end3, "緊急事態宣言"] = list(range(1, (end3 - start3).days + 2))
df.to_csv("./data/emergency.csv")