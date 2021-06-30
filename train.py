import warnings
import pickle
import argparse
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna

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
args = parser.parse_args()
net_name_list = args.net_list
if not net_name_list:
    raise ValueError("Arguments must be passed")
repeat = args.repeat
patience = args.patience

exist_net = set([os.path.basename(i)[:-3] for i in glob.glob("./nets/*.py")])
for net_name in net_name_list:
    if net_name not in exist_net:
        raise ValueError(f"{net_name} does not exist")

net_dict = nets.get_nets(net_name_list)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_val(Net, kwargs, dataloader_dict):
    net = Net(**kwargs)
    early_stopping = EarlyStopping(patience)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    break_flag = False
    pbar = tqdm(range(30000), leave=False)
    for _ in pbar:
        for phase in ["train", "val"]:
            dataloader = dataloader_dict[phase]
            if phase == "train":
                net.train()
            else:
                net.eval()
            epoch_loss = 0
            epoch_mae = 0
            with torch.set_grad_enabled(phase == "train"):
                for x, t in dataloader:
                    x = x.to(DEVICE)
                    t = t.to(DEVICE)
                    optimizer.zero_grad()
                    y = net(x)
                    loss = torch.sqrt(criterion(y, t))
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item()*x.size(0)
                    y2 = inverse_standard(y.detach())
                    t2 = inverse_standard(t.detach())
                    epoch_mae += F.l1_loss(y2, t2, reduction="sum").item()
            data_len = len(dataloader.dataset)
            epoch_loss /= data_len
            epoch_mae /= data_len
            if phase == "val":
                break_flag = early_stopping(net, epoch_mae)
                pbar.set_postfix(epoch_mae=f"{epoch_mae:.1f}")
        if break_flag:
            break
        state_dict = early_stopping.state_dict
        net.load_state_dict(state_dict)
    return net, epoch_mae

for net_name, Net in net_dict.items():
    tqdm.write(f"【{net_name}】")

    data, normalization_idx = Net.get_data()
    net_params = Net.net_params
    train_val_test = TrainValTest(data, normalization_idx)
    dataloader_dict = train_val_test.dataloader_dict
    inverse_standard = train_val_test.inverse_standard

    def objective(trial):
        kwargs = {}
        for net_param in net_params:
            x = trial.suggest_categorical(*net_param)
            kwargs[net_param[0]] = x
        _, val_mae = train_val(Net, kwargs, dataloader_dict)
        return val_mae

    study = optuna.create_study()
    study.optimize(objective, n_trials=2)

    best_params = study.best_params
    tqdm.write(str(best_params))
    net, epoch_mae = train_val(Net, best_params, dataloader_dict)
    torch.save(net.to("cpu").state_dict(), f"./result_pth/{net_name}.pth")
