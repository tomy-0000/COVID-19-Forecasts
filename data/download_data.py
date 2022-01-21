#%%
import pandas as pd

df = (
    pd.read_csv("https://covid19.mhlw.go.jp/public/opendata/newly_confirmed_cases_daily.csv", parse_dates=True)
    .rename(columns={"Date": "date"})
    .set_index("date", drop=True)
)
df.to_csv("data/use/Japan_count.csv")
df["Tokyo"].to_csv("data/use/Tokyo_count.csv")

df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
df.to_csv("data/use/World_features.csv")
