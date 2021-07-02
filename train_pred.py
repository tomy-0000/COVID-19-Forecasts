import argparse
import glob
import os
import re
import json
import warnings
warnings.simplefilter("ignore")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
import torch
import torch.nn as nn
import torch.nn.functional as F
if "get_ipython" in globals():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
from utils import TrainValTest, EarlyStopping
import nets

parser = argparse.ArgumentParser()
parser.add_argument("net_list", nargs="*")
parser.add_argument('--repeat', default=100, type=int)
parser.add_argument('--patience', default=500, type=int)
parser.add_argument('--n_trials', default=100, type=int)
args = parser.parse_args()
net_name_list = args.net_list
if not net_name_list:
    raise ValueError("Arguments must be passed")
repeat = args.repeat
patience = args.patience
n_trials = args.n_trials

exist_net_name_list = [os.path.basename(i)[:-3] for i in glob.glob("./nets/*.py")]
tmp_net_name_set = set()
for net_name in net_name_list:
    net_name = net_name + "$"
    for exist_net_name in exist_net_name_list:
        if re.search(net_name, exist_net_name):
            tmp_net_name_set.add(exist_net_name)
net_name_list = list(tmp_net_name_set)
tqdm.write(str(net_name_list))

net_dict = nets.get_nets(net_name_list)

with open("./best_params_dict.json") as f:
    best_params_dict = json.load(f)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def train_val(Net, kwargs, dataloader_dict, inverse_standard, tqdm_pos):
    net = Net(**kwargs).to(DEVICE)
    early_stopping = EarlyStopping(patience)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    break_flag = False
    pbar3 = tqdm(range(30000), leave=False, position=tqdm_pos)
    pbar3.set_description("train_val")
    for _ in pbar3:
        for phase in ["train", "val"]:
            dataloader = dataloader_dict[phase]
            if phase == "train":
                net.train()
            else:
                net.eval()
            with torch.set_grad_enabled(phase == "train"):
                for x, t in dataloader:
                    x = x.to(DEVICE)
                    t = t.to(DEVICE)
                    optimizer.zero_grad()
                    y = net(x).unsqueeze(-1)
                    loss = torch.sqrt(criterion(y, t))
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    y2 = inverse_standard(y.detach())
                    t2 = inverse_standard(t.detach())
                    epoch_mae = F.l1_loss(y2, t2).item()
            if phase == "val":
                break_flag = early_stopping(net, epoch_mae)
                pbar3.set_postfix(epoch_mae=f"{epoch_mae:.1f}")
        if break_flag:
            break
        state_dict = early_stopping.state_dict
        net.load_state_dict(state_dict)
    return net, epoch_mae

def test(net, dataset_dict, inverse_standard):
    net = net.to("cpu")
    net.eval()
    with torch.set_grad_enabled(False):
        for phase in ["train", "test"]:
            dataset = dataset_dict[phase]
            x, t = dataset[0]
            x = x.unsqueeze(0)
            y = net(x).numpy().reshape(-1)
            t = t.numpy().reshape(-1)
            y = inverse_standard(y)
            t = inverse_standard(t)
            mae = sum(abs(y - t))/len(y)

            plt.figure(figsize=(12, 8))
            x = np.arange(len(y))
            sns.lineplot(x=x, y=y, label="predict", linewidth=4)
            sns.lineplot(x=x, y=t, label="gt", linewidth=4)
            plt.title(f"pred_{phase} (mae:{mae:.3f})")
            plt.legend(fontsize=16)
            plt.savefig(f"./result_img/{net_name}_{phase}_pred.png")


pbar1 = tqdm(net_dict.items(), position=0)
for net_name, Net in pbar1:
    pbar1.set_description(net_name)

    data, normalization_idx = Net.get_data()
    net_params = Net.net_params
    train_val_test = TrainValTest(data, normalization_idx)
    dataloader_dict = train_val_test.dataloader_dict
    inverse_standard = train_val_test.inverse_standard
    predict_seq = train_val_test.predict_seq

    def objective(trial):
        kwargs = {"predict_seq": predict_seq}
        for net_param in net_params:
            x = trial.suggest_categorical(*net_param)
            kwargs[net_param[0]] = x
        _, val_mae = train_val(Net, kwargs, dataloader_dict, inverse_standard, 2)
        global pbar2
        pbar2.update()
        return val_mae

    pbar2 = tqdm(total=n_trials, leave=False, position=1)
    pbar2.set_description("objective")
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)

    best_value = study.best_value
    best_params = study.best_params
    tqdm.write(f"【{net_name}】 best value: {int(best_value)}, best params: {best_params}")
    best_params_dict[net_name] = best_params
    best_params = best_params.copy()
    best_params["predict_seq"] = predict_seq
    net, epoch_mae = train_val(Net, best_params, dataloader_dict, inverse_standard, 1)
    dataset_dict = train_val_test.dataset_dict
    test(net, dataset_dict, inverse_standard)
    with open("./best_params_dict.json", "w") as f:
        json.dump(best_params_dict, f, indent=2)
