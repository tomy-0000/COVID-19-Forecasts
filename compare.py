#%%
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("result_list", nargs="*")
result_name_list = parser.parse_args().result_list
if not result_name_list:
    raise ValueError("Arguments must be passed")

result_path_list = [f"./result_pkl/{s}.pkl" for s in result_name_list]
result_arr_list = []
for result_path in result_path_list:
    with open(result_path, "rb") as f:
        result_arr_list.append(pickle.load(f))

fig, ax = plt.subplots()
sns.boxplot(data=result_arr_list, whis=[0, 100])
sns.stripplot(data=result_arr_list, size=4, color="white", edgecolor="black", linewidth=1)
ax.set_xticklabels(result_name_list)
ax.set_ylabel("mae")
plt.savefig(f"./result_img/boxplot_{'_'.join(result_name_list)}.png")
