#%%
import pandas as pd

print("Downloading Japan")
df = pd.read_csv("https://covid19.mhlw.go.jp/public/opendata/newly_confirmed_cases_daily.csv")
df.to_csv("data/raw/Japan.csv", index=False)

print("Downloading World")
df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
df.to_csv("data/raw/World.csv", index=False)
