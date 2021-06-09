#%%
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
if "get_ipython" in globals():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import utils
import config
import nets

#%%
mae_list = []
net_name_list = []

data = nets.net1.get_data()
Net = nets.net1.Net1
net_config = config.net1_config
train_val_test = utils.TrainValTest(data, **config.dataset_config)
net_name = Net.__name__
tqdm.write(f"【{net_name}】")
net_name_list.append(net_name)
utils.run(Net, net_config, train_val_test, config.epoch,
          patience=config.patience)
result = utils.run_repeatedly(Net, net_config, train_val_test, config.epoch,
                              patience=config.patience,
                              repeat_num=config.repeat_num)
mae_list.append(result)

data = nets.net2.get_data()
Net = nets.net2.Net2
net_config = config.net2_config
train_val_test = utils.TrainValTest(data, **config.dataset_config)
net_name = Net.__name__
tqdm.write(f"【{net_name}】")
net_name_list.append(net_name)
utils.run(Net, net_config, train_val_test, config.epoch,
          patience=config.patience)
result = utils.run_repeatedly(Net, net_config, train_val_test, config.epoch,
                              patience=config.patience,
                              repeat_num=config.repeat_num)
mae_list.append(result)

data = nets.net3.get_data()
Net = nets.net3.Net3
net_config = config.net3_config
train_val_test = utils.TrainValTest(data, **config.dataset_config)
net_name = Net.__name__
tqdm.write(f"【{net_name}】")
net_name_list.append(net_name)
utils.run(Net, net_config, train_val_test, config.epoch,
          patience=config.patience)
result = utils.run_repeatedly(Net, net_config, train_val_test, config.epoch,
                              patience=config.patience,
                              repeat_num=config.repeat_num)
mae_list.append(result)

result_df = pd.DataFrame({i: j for i, j in zip(net_name_list, mae_list)})
with open('result_df.pkl', 'wb') as f:
    pickle.dump(result_df, f)
plt.figure()
sns.boxplot(data=result_df)
sns.swarmplot(data=result_df, color="white", size=7, edgecolor="black", linewidth=2)
plt.savefig("./result_img/boxplot.png")
