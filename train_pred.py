#%%
import pickle
if "get_ipython" in globals():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import utils
import config
import nets

#%%
data = nets.net4.get_data()
Net = nets.net4.Net4
net_config = config.net1_config
train_val_test = utils.TrainValTest(data, **config.dataset_config)
net_name = Net.__name__
tqdm.write(f"【{net_name}】")
utils.run(Net, net_config, train_val_test, config.epoch,
          patience=config.patience)
result = utils.run_repeatedly(Net, net_config, train_val_test, config.epoch,
                              patience=config.patience,
                              repeat_num=config.repeat_num)
with open(f'./result/{net_name}.pkl', 'wb') as f:
    pickle.dump(result, f)