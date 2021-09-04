#%%
import datetime
import glob
import pandas as pd

#%%
# 感染者数
url = f"https://docs.google.com/spreadsheets/d/10MFfRQTblbOpuvOs_yjIYgntpMGBg592dL8veXoPpp4/gviz/tq?tqx=out:csv&sheet=%E7%BD%B9%E6%82%A3%E8%80%85%E7%B5%B1%E8%A8%88"
df = pd.read_csv(url)
df = df.set_index("日付", drop=True)

for i in df:
    tmp = df[i].astype(str)
    tmp = tmp.str.replace(",", "")
    df[i] = pd.to_numeric(tmp, errors="coerce")
df.to_csv("./data_raw/count.csv")

#%%
df = pd.read_csv("./data_raw/count.csv", index_col=0)
df.loc["2020/06/01":"2021/05/31"].to_csv("./data_use/count.csv")

#%%
# 天気
csv_list = sorted(glob.glob("./data_raw/weather_raw/202*.csv"))
df = pd.DataFrame()
usecols = [0, 1, 4, 8, 13, 16, 19, 22, 25]
names = ["日付", "気温", "降水量", "風速", "現地気圧", "相対湿度", "蒸気圧", "天気", "雲量"]
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
df.to_csv("./data_raw/weather.csv")

#%%
df = pd.read_csv("./data_raw/weather.csv", index_col=0)
df.loc["2020-06-01":"2021-05-31"].to_csv("./data_use/wheather.csv")

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

#%%
df = pd.read_csv("./data_raw/emergency.csv", index_col=0)
df.loc["2020-06-01":"2021-05-31"].to_csv("./data_use/emergency.csv")
