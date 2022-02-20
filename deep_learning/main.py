import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

import nets
from utils import DEVICE, EarlyStopping, get_dataloader, inverse_scaler

sns.set()
torch.manual_seed(555)
torch.backends.cudnn.benchmark = False


def train(net, optimizer, dataloader, scaler):
    net.train()
    rmse = 0.0
    mae = 0.0
    for enc_X, dec_X, t, location in dataloader:
        enc_X = enc_X.to(DEVICE)
        dec_X = dec_X.to(DEVICE)
        t = t.to(DEVICE)
        optimizer.zero_grad()
        y = net(enc_X, dec_X)
        batch_loss = F.mse_loss(y, t) ** 0.5
        batch_loss.backward()
        optimizer.step()
        if scaler is None:
            rmse += ((y - t) ** 2).sum().detach().item()
            mae += F.l1_loss(y, t, reduction="sum").detach().item()
        else:
            location_num = dataloader.dataset.location_num
            y_inverse = inverse_scaler(y, location, location_num, scaler)
            t_inverse = inverse_scaler(t, location, location_num, scaler)
            rmse += ((y_inverse - t_inverse) ** 2).sum()
            mae += abs(y_inverse - t_inverse).sum()
    rmse = (rmse / dataloader.dataset.size) ** 0.5
    mae = mae / dataloader.dataset.size
    return rmse, mae


def val_test(net, dataloader, scaler, is_test):
    net.eval()
    rmse = 0.0
    mae = 0.0
    y_all = []
    t_all = []
    with torch.no_grad():
        for enc_X, dec_X, t, location in dataloader:
            enc_X = enc_X.to(DEVICE)
            dec_X = dec_X.to(DEVICE)
            t = t.to(DEVICE)
            if is_test:
                y = net.test(enc_X, dec_X, t.shape[-1])
            else:
                y = net(enc_X, dec_X)
            if scaler is None:
                rmse += ((y - t) ** 2).sum().item()
                mae += F.l1_loss(y, t, reduction="sum").detach().item()
                y_all += y.cpu().numpy().reshape(-1).tolist()
                t_all += t.cpu().numpy().reshape(-1).tolist()
            else:
                location_num = dataloader.dataset.location_num
                y_inverse = inverse_scaler(y, location, location_num, scaler)
                t_inverse = inverse_scaler(t, location, location_num, scaler)
                rmse += ((y_inverse - t_inverse) ** 2).sum()
                mae += abs(y_inverse - t_inverse).sum()
                y_all += y_inverse.tolist()
                t_all += t_inverse.tolist()
    rmse = (rmse / dataloader.dataset.size) ** 0.5
    mae = mae / dataloader.dataset.size
    if is_test:
        corrcoef = np.corrcoef(np.array([y_all, t_all]))[0][1]
    else:
        corrcoef = None
    return rmse, mae, corrcoef


def run(train_dataloader, val_dataloader, total_epoch, patience, batch_size, net_name, scaler):
    train_loss_list = []
    train_mae_list = []
    val_loss_list = []
    val_mae_list = []
    early_stopping = EarlyStopping(patience)
    if net_name == "transformer":
        # optuna
        net = nets.TransformerNet(
            d_model=84,
            nhead=1,
            num_encoder_layers=5,
            num_decoder_layers=1,
            dim_feedforward=2048,
            dropout=0.6996650500967093,
            batch_size=batch_size,
        ).to(DEVICE)
    elif net_name == "lstm":
        # optuna
        net = nets.LSTMNet(d_model=459, num_layers=5, dropout=0.6722579147817102, bidirectional=True).to(DEVICE)
    optimizer = optim.AdamW(net.parameters(), lr=1e-5)
    pbar = tqdm(total=total_epoch, position=0)
    desc = tqdm(total=total_epoch, position=1, bar_format="{desc}", desc="")
    for epoch in range(total_epoch):
        train_loss, train_mae = train(net, optimizer, train_dataloader, scaler)
        train_loss_list.append(train_loss)
        train_mae_list.append(train_mae)
        val_loss, val_mae, _ = val_test(net, val_dataloader, scaler, is_test=False)
        val_loss_list.append(val_loss)
        val_mae_list.append(val_mae)
        pbar.update(1)
        if early_stopping(net, val_loss):
            break
        desc_str = f"Train RMSE: {train_loss:.3f} | Val RMSE: {val_loss:.3f} | Train MAE: {train_mae:.3f} | Val MAE: {val_mae:.3f} | Best Val RMSE: {early_stopping.best_value:.3f} | EaryStopping Counter: {early_stopping.counter}/{early_stopping.patience}"
        desc.set_description(desc_str)
    net.load_state_dict(early_stopping.best_state_dict)
    return train_loss_list, val_loss_list, train_mae_list, val_mae_list, net


def plot_history(train_loss_list, val_loss_list, train_mae_list, val_mae_list):
    plt.figure()
    plt.plot(train_loss_list, label="train")
    plt.plot(val_loss_list, label="val")
    plt.legend()
    plt.title("loss")
    plt.savefig("deep_learning/result/loss.png")
    plt.figure()
    plt.plot(train_mae_list, label="train")
    plt.plot(val_mae_list, label="val")
    plt.legend()
    plt.title("mae")
    plt.savefig("deep_learning/result/mae.png")


def plot_predict(net, dataloader, location2id, scaler, mode):
    X, _, t, _ = dataloader.dataset[0]
    X_seq = len(X)
    t_seq = len(t)
    y_all_location = []
    t_all_location = []
    for location_str, location_id in tqdm(location2id.items(), leave=False):
        net.eval()
        y_each_location = []
        t_each_location = []
        cnt = 0
        is_first = True
        with torch.no_grad():
            for enc_X, dec_X, t, location in dataloader:
                enc_X = enc_X[location == location_id]
                dec_X = dec_X[location == location_id]
                t = t[location == location_id]
                location = location[location == location_id]
                location_num = dataloader.dataset.location_num
                if len(t) == 0:
                    continue
                if is_first:
                    if scaler is None:
                        t_each_location += enc_X[[0]].numpy().reshape(-1).tolist()
                    else:
                        t_each_location += inverse_scaler(enc_X[[0]], location[[0]], location_num, scaler).tolist()
                    is_first = False
                enc_X = enc_X.to(DEVICE)
                dec_X = dec_X.to(DEVICE)
                t = t.to(DEVICE)
                y = net.test(enc_X, dec_X, t.shape[-1])
                y_all_location += y.cpu().numpy().reshape(-1).tolist()
                t_all_location += t.cpu().numpy().reshape(-1).tolist()
                if scaler is None:
                    y_each_location += y.cpu().numpy().reshape(-1).tolist()
                    t_each_location += t.cpu().numpy().reshape(-1).tolist()
                else:
                    y_each_location += inverse_scaler(y, location, location_num, scaler).tolist()
                    t_each_location += inverse_scaler(t, location, location_num, scaler).tolist()
                cnt += len(y_each_location)
        mae = abs(np.array(y_each_location) - np.array(t_each_location[X_seq:])).sum()
        mae /= cnt
        fig, ax = plt.subplots()
        for i in range(0, len(y_each_location), t_seq):
            if i > 0:
                ax.plot(range(X_seq + i - 1, X_seq + i + 1), y_each_location[i - 1 : i + 1], color="C0", linestyle="--")
            ax.plot(range(X_seq + i, X_seq + i + t_seq), y_each_location[i : i + t_seq], color="C0")
        ax.lines[0].set_label("predict")
        ax.plot(t_each_location, label="ground truth", color="C1")
        ax.legend()
        ax.set_title(f"{location_str}_{mae:.1f}.png")
        fig.savefig(f"deep_learning/result/{mode}/{location_str}.png")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["World", "Japan", "Tokyo"], default="World")
    parser.add_argument("--X_seq", type=int, default=10)
    parser.add_argument("--t_seq", type=int, default=4)
    parser.add_argument("--total_epoch", type=int, default=100000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_inverse", action="store_true")
    parser.add_argument("--net", default="transformer")
    args = parser.parse_args()

    train_dataloader, val_dataloader, test_dataloader, scaler, location2id = get_dataloader(
        X_seq=args.X_seq, t_seq=args.t_seq, use_val=True, mode=args.mode, batch_size=args.batch_size
    )
    if not args.use_inverse:
        scaler = None

    train_loss_list, val_loss_list, train_mae_list, val_mae_list, net = run(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        total_epoch=args.total_epoch,
        patience=args.patience,
        batch_size=args.batch_size,
        net_name=args.net,
        scaler=scaler,
    )
    plot_history(train_loss_list, val_loss_list, train_mae_list, val_mae_list)

    train_loss, train_mae, corrcoef = val_test(net, train_dataloader, scaler, is_test=True)
    tqdm.write(f"Train RMSE: {train_loss:.3f} | Train MAE: {train_mae:.3f} | Train Corr Coef: {corrcoef:.3f}")
    test_loss, test_mae, corrcoef = val_test(net, test_dataloader, scaler, is_test=True)
    tqdm.write(f"Test RMSE: {test_loss:.3f} | Test MAE: {test_mae:.3f} | Test Corr Coef: {corrcoef:.3f}")

    plot_predict(net, train_dataloader, location2id, scaler, os.path.join(args.mode, "train"))
    plot_predict(net, test_dataloader, location2id, scaler, args.mode)
