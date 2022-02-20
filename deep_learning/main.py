import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import optuna
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


def val(net, dataloader, scaler, transform):
    net.eval()
    rmse = 0.0
    mae = 0.0
    with torch.no_grad():
        for enc_X, dec_X, t, location in dataloader:
            enc_X = enc_X.to(DEVICE)
            dec_X = dec_X.to(DEVICE)
            t = t.to(DEVICE)
            y = net(enc_X, dec_X)
            if scaler is None:
                rmse += ((y - t) ** 2).sum().item()
                mae += F.l1_loss(y, t, reduction="sum").detach().item()
            else:
                location_num = dataloader.dataset.location_num
                y_inverse = inverse_scaler(y, location, location_num, scaler)
                t_inverse = inverse_scaler(t, location, location_num, scaler)
                rmse += ((y_inverse - t_inverse) ** 2).sum()
                mae += abs(y_inverse - t_inverse).sum()
    rmse = (rmse / dataloader.dataset.size) ** 0.5
    mae = mae / dataloader.dataset.size
    return loss, mae


def test(net, dataloader, scaler, transform):
    net.eval()
    loss = 0.0
    mae = 0.0
    with torch.no_grad():
        for enc_X, dec_X, t, location in dataloader:
            enc_X = enc_X.to(DEVICE)
            dec_X = dec_X.to(DEVICE)
            t = t.to(DEVICE)
            y = net.test(enc_X, dec_X, t.shape[-1])
            loss += F.mse_loss(y, t, reduction="sum").detach().item()
            if transform == "scaled":
                mae += F.l1_loss(y, t, reduction="sum").detach().item()
            else:
    return rmse, mae, corrcoef


def run(
    train_dataloader,
    val_dataloader,
    total_epoch,
    patience,
    mode,
    batch_size,
    net_name="transformer",
):
    torch.manual_seed(555)

    train_loss_list = []
    train_mae_list = []
    val_loss_list = []
    val_mae_list = []
    early_stopping = EarlyStopping(patience)
    tqdm.write(f"[{mode}]")
    if net_name == "transformer":
        net = nets.TransformerNet(
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.6,
            batch_size=batch_size,
        ).to(DEVICE)
        # net = nets.TransformerNet(batch_size=batch_size, **params_list).to(DEVICE)
    elif net_name == "lstm":
        net = nets.LSTMNet(d_model=512, num_layers=1).to(DEVICE)
    optimizer = optim.AdamW(net.parameters(), lr=1e-5)
    pbar = tqdm(total=total_epoch, position=0)
    desc = tqdm(total=total_epoch, position=1, bar_format="{desc}", desc="")
    for epoch in range(total_epoch):
        train_loss, train_mae = train(net, optimizer, train_dataloader, scaler)
        train_loss_list.append(train_loss)
        train_mae_list.append(train_mae)
        if mode == "Train And Val":
            val_loss, val_mae = val(net, val_dataloader, scaler)
            val_loss_list.append(val_loss)
            val_mae_list.append(val_mae)
        if mode == "Leak":
            val_loss, val_mae = test(net, val_dataloader, scaler)
            val_loss_list.append(val_loss)
            val_mae_list.append(val_mae)
        pbar.update(1)
        if val_dataloader is not None:
            desc_str = f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Train MAE: {train_mae:.3f} | Val MAE: {val_mae:.3f} | Best Val Loss: {early_stopping.best_value2:.3f} | EaryStopping Counter: {early_stopping.counter}/{early_stopping.patience}"
        else:
            desc_str = f"Train Loss: {train_loss:.3f} | Train MAE: {train_mae:.3f}"
        desc.set_description(desc_str)
        if val_dataloader is not None:
            if early_stopping(net, val_loss, val_mae):
                break
    if val_dataloader is not None:
        net.load_state_dict(early_stopping.state_dict)
    epoch = max(5, epoch - patience)
    return epoch, train_loss_list, val_loss_list, train_mae_list, val_mae_list, net


def plot_history(train_loss_list, val_loss_list, train_mae_list, val_mae_list, suffix):
    plt.figure()
    plt.plot(train_loss_list, label="train")
    if val_loss_list is not None:
        plt.plot(val_loss_list, label="val")
    plt.legend()
    plt.title("loss")
    plt.savefig(f"deep_learning/result/loss_{suffix}.png")
    plt.figure()
    plt.plot(train_mae_list, label="train")
    if val_mae_list is not None:
        plt.plot(val_mae_list, label="val")
    plt.legend()
    plt.title("mae")
    plt.savefig(f"deep_learning/result/mae_{suffix}.png")


def plot_predict(net, dataloader, location2id, scaler, mode, suffix):
    X, _, t, _ = dataloader.dataset[0]
    X_seq = len(X)
    t_seq = len(t)
    for location_str, location_id in tqdm(location2id.items(), leave=False):
        net.eval()
        y_inverse = []
        t_inverse = []
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
                    t_inverse += inverse_scaler(enc_X[[0]], location[[0]], location_num, scaler).tolist()
                    is_first = False
                enc_X = enc_X.to(DEVICE)
                dec_X = dec_X.to(DEVICE)
                t = t.to(DEVICE)
                if "/train" in mode:
                    y = net(enc_X, dec_X)
                else:
                    y = net.test(enc_X, dec_X, t.shape[-1])
                y_inverse += inverse_scaler(y, location, location_num, scaler).tolist()
                t_inverse += inverse_scaler(t, location, location_num, scaler).tolist()
                cnt += len(y_inverse)
        mae = abs(np.array(y_inverse) - np.array(t_inverse[X_seq:])).sum()
        mae /= cnt
        fig, ax = plt.subplots()
        for i in range(0, len(y_inverse), t_seq):
            if i > 0:
                ax.plot(range(X_seq + i - 1, X_seq + i + 1), y_inverse[i - 1 : i + 1], color="C0", linestyle="--")
            ax.plot(range(X_seq + i, X_seq + i + t_seq), y_inverse[i : i + t_seq], color="C0")
        ax.lines[0].set_label("predict")
        ax.plot(t_inverse, label="ground truth", color="C1")
        ax.legend()
        ax.set_title(f"{location_str}_{mae:.1f}.png")
        fig.savefig(f"deep_learning/result/{mode}/{location_str}_{suffix}.png")
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
    args = parser.parse_args()

    train_dataloader, val_dataloader, test_dataloader, scaler, location2id = get_dataloader(
        args.X_seq, args.t_seq, True, args.mode, args.batch_size
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
