#%%
import pandas as pd
import utils
import config
import nets

url1 ="https://docs.google.com/spreadsheets/d/1Ot0T8_YZ2Q0dORnKEhcUmuYCqZ1y81PIsIAMB7WZE8g/gviz/tq?tqx=out:csv&sheet=%E7%BD%B9%E6%82%A3%E8%80%85_%E6%9D%B1%E4%BA%AC_2020"
url2 = "https://docs.google.com/spreadsheets/d/1V1eJM1mupE9gJ6_k0q_77nlFoRuwDuBliMLcMdDMC_E/gviz/tq?tqx=out:csv&sheet=%E7%BD%B9%E6%82%A3%E8%80%85_%E6%9D%B1%E4%BA%AC_2021"

df = pd.concat([pd.read_csv(url1), pd.read_csv(url2)])
df = df[["公表日", "年代", "性別"]]
df = df[df["公表日"].notna()]
start = df["公表日"].iat[0]
end = df["公表日"].iat[-1]
index = pd.date_range(start=start, end=end)
df2 = pd.DataFrame(0, columns=["count"], index=index)
for date, tmp_df in df.groupby("公表日"):
    df2.loc[date, "count"] += len(tmp_df)

#%%
data1 = nets.net1.get_data(df2)
train_val_test = utils.TrainValTest(data1, **config.dataset_config)
Net1 = nets.net1.Net1
utils.run(Net1, config.net1_config, train_val_test, config.epoch,
          patience=500)
utils.run_repeatedly(Net1, config.net1_config, train_val_test, config.epoch,
                     patience=config.patience)