#%%
import pickle
import argparse
import glob
import os

from torch._C import Value
if "get_ipython" in globals():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import utils
from config import Config
import nets

parser = argparse.ArgumentParser()
parser.add_argument("net_list", nargs="*")
net_name_list = parser.parse_args().net_list
if not net_name_list:
    raise ValueError("Arguments must be passed")

exist_net = set([os.path.basename(i)[:-3] for i in glob.glob("./nets/*.py")])
for net_name in net_name_list:
    if net_name not in exist_net:
        raise ValueError(f"{net_name} does not exist")

net_dict = nets.get_nets(net_name_list)

for net_name, Net in net_dict.items():
    data = Net.get_data()
    dataset_config = Net.dataset_config
    net_config = Net.net_config
    train_val_test = utils.TrainValTest(data, **dataset_config)

    tqdm.write(f"【{net_name}】")
    utils.run(Net, net_name, net_config, train_val_test, epoch=Config.epoch, use_best=True,
              plot=True, log=True, patience=Config.patience)
    result = utils.run_repeatedly(Net, net_name, net_config, train_val_test, epoch=Config.epoch,
                                  lr=Config.lr, patience=Config.patience,
                                  repeat_num=Config.repeat_num)
    with open(f'./result_pkl/{net_name}.pkl', 'wb') as f:
        pickle.dump(result, f)
