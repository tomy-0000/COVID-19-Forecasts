#%%
import os
import pickle
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
result_list = []
net_name_list = []
result_file_list = glob.glob("./result/*.pkl")
for result_file in result_file_list:
    with open(result_file, "rb") as f:
        result_list.append(pickle.load(f))
    net_name_list.append(os.path.basename(result_file)[:-4])

result_df = pd.DataFrame({i: j for i, j in zip(net_name_list, result_list)})
plt.figure()
sns.boxplot(data=result_df)
sns.swarmplot(data=result_df, color="white", size=7, edgecolor="black", linewidth=2)
plt.savefig("./result_img/boxplot.png")
