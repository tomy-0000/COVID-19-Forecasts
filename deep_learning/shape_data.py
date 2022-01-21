#%%
import datetime
import glob
import pandas as pd

#%%
# 感染者数
count_df = pd.read_csv("data/raw/count.csv", index_col=0, parse_dates=True)
count_df = count_df[count_df["Prefecture"] == "Tokyo"].drop("Prefecture", axis=1)
count_df = count_df.rename(columns={"Newly confirmed cases": "count"})
count_df.to_csv("data/use/count.csv")

#%%
# 天気
csv_list = sorted(glob.glob("data/raw/weather/*.csv"))
weather_df = pd.DataFrame()
usecols = [0, 1, 4, 8, 13, 16, 19, 22, 25]
names = ["日付", "気温", "降水量", "風速", "現地気圧", "相対湿度", "蒸気圧", "天気", "雲量"]
for csv in csv_list:
    tmp_df = pd.read_csv(
        csv, encoding="shift-jis", skiprows=3, usecols=usecols, index_col=0, names=names
    ).iloc[3:, :]
    tmp_df.index = pd.to_datetime(tmp_df.index)
    use = (7 <= tmp_df.index.hour) & (tmp_df.index.hour <= 15)
    tmp_df = tmp_df[use].fillna(method="bfill")
    weather_df = pd.concat([weather_df, tmp_df])
weather_df["雲量"] = weather_df["雲量"].str.extract("(\d+)")
dtypes = [float, float, float, float, int, float, int, int]
weather_df = weather_df.astype({i: j for i, j in zip(names[1:], dtypes)})
weather_df["天気"] = weather_df["天気"].map(
    {j: i for i, j in enumerate(sorted(weather_df["天気"].unique()))}
)
weather_df = weather_df.resample("D").mean()
weather_df.to_csv("data/use/weather.csv")

#%%
# 緊急事態宣言
index = pd.date_range(
    start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2021, 12, 31)
)
emergency_df = pd.DataFrame(0, columns=["緊急事態宣言"], index=index)
start1 = datetime.datetime(2020, 4, 7)
end1 = datetime.datetime(2020, 5, 25)
emergency_df.loc[start1:end1, "緊急事態宣言"] = list(range(1, (end1 - start1).days + 2))
start2 = datetime.datetime(2021, 1, 8)
end2 = datetime.datetime(2021, 3, 21)
emergency_df.loc[start2:end2, "緊急事態宣言"] = list(range(1, (end2 - start2).days + 2))
start3 = datetime.datetime(2021, 4, 25)
end3 = datetime.datetime(2021, 5, 31)
emergency_df.loc[start3:end3, "緊急事態宣言"] = list(range(1, (end3 - start3).days + 2))
emergency_df.to_csv("data/use/emergency.csv")
