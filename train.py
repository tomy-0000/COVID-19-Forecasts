import warnings
warnings.filterwarnings('ignore')
import pickle
import argparse
import glob
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import early_stopping
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import optuna

if "get_ipython" in globals():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import utils
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

for net_name, Net in net_dict.items():
    tqdm.write(f"【{net_name}】")

    data, normalization_idx = Net.get_data()
    net_params = Net.net_params
    train_val_test = utils.TrainValTest(data, normalization_idx)
    dataloader_dict = train_val_test.dataloader_dict
    inverse_standard = train_val_test.inverse_standard
    early_stopping = EarlyStopping(monitor="val_loss", patience=500)
    trainer = pl.Trainer(callbacks=[early_stopping], max_epochs=30000)

    def objective(trial):
        kwargs = {"inverse_standard": inverse_standard}
        for net_param in net_params:
            x = trial.suggest_categorical(*net_param)
            kwargs[net_param[0]] = x
        net = Net(**kwargs)
        trainer = pl.Trainer(callbacks=[early_stopping], max_epochs=30000)
        trainer.fit(net, dataloader_dict["train"],
        dataloader_dict["val"])
        val_acc = trainer.callback_metrics["val_acc"]
        return val_acc

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

        # with open(f'./result_pkl/{net_name}.pkl', 'wb') as f:
        #     pickle.dump(result, f)
    best_params = study.best_params
    print(best_params)