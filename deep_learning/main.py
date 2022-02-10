import argparse

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
torch.manual_seed(111)
torch.backends.cudnn.benchmark = False


def train(net, optimizer, dataloader, scaler):
    net.train()
    loss = 0.0
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
        loss += F.mse_loss(y, t, reduction="sum").detach().item()
        location_num = dataloader.dataset.location_num
        y_inverse = inverse_scaler(y, location, location_num, scaler)
        t_inverse = inverse_scaler(t, location, location_num, scaler)
        mae += abs(y_inverse - t_inverse).sum()
    loss = (loss / dataloader.dataset.size) ** 0.5
    mae = mae / dataloader.dataset.size
    return loss, mae


def val(net, dataloader, scaler):
    net.eval()
    loss = 0.0
    mae = 0.0
    with torch.no_grad():
        for enc_X, dec_X, t, location in dataloader:
            enc_X = enc_X.to(DEVICE)
            dec_X = dec_X.to(DEVICE)
            t = t.to(DEVICE)
            y = net(enc_X, dec_X)
            loss += F.mse_loss(y, t, reduction="sum").detach().item()
            location_num = dataloader.dataset.location_num
            y_inverse = inverse_scaler(y, location, location_num, scaler)
            t_inverse = inverse_scaler(t, location, location_num, scaler)
            mae += abs(y_inverse - t_inverse).sum()
    loss = (loss / dataloader.dataset.size) ** 0.5
    mae = mae / dataloader.dataset.size
    return loss, mae


def test(net, dataloader, scaler):
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
            location_num = dataloader.dataset.location_num
            y_inverse = inverse_scaler(y, location, location_num, scaler)
            t_inverse = inverse_scaler(t, location, location_num, scaler)
            mae += abs(y_inverse - t_inverse).sum()
    loss = (loss / dataloader.dataset.size) ** 0.5
    mae = mae / dataloader.dataset.size
    return loss, mae


def run(train_dataloader, val_dataloader, total_epoch, patience, mode, batch_size):
    train_loss_list = []
    train_mae_list = []
    val_loss_list = []
    val_mae_list = []
    early_stopping = EarlyStopping(patience)
    tqdm.write(f"[{mode}]")
    # net = nets.TransformerNet(
    #     d_model=512,
    #     nhead=8,
    #     num_encoder_layers=6,
    #     num_decoder_layers=6,
    #     dim_feedforward=2048,
    #     dropout=0.1,
    #     batch_size=batch_size,
    # ).to(DEVICE)
    net = nets.LSTMNet(d_model=512).to(DEVICE)
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
            desc_str = f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Train MAE: {train_mae:.3f} | Val MAE: {val_mae:.3f} | Best Val Loss: {early_stopping.best_value:.3f} | EaryStopping Counter: {early_stopping.counter}/{early_stopping.patience}"
        else:
            desc_str = f"Train Loss: {train_loss:.3f} | Train MAE: {train_mae:.3f}"
        desc.set_description(desc_str)
        if val_dataloader is not None:
            if early_stopping(net, val_loss):
                break
    epoch -= patience
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
    # len(dataloader) == 1の時だけ
    X, _, t, _ = dataloader.dataset[0]
    X_seq = len(X)
    t_seq = len(t)
    for location_str, location_id in tqdm(location2id.items(), leave=False):
        net.eval()
        y_inverse = []
        t_inverse = []
        cnt = 0
        with torch.no_grad():
            for i, (enc_X, dec_X, t, location) in enumerate(dataloader):
                enc_X = enc_X[location == location_id]
                dec_X = dec_X[location == location_id]
                t = t[location == location_id]
                location = location[location == location_id]
                location_num = dataloader.dataset.location_num
                if i == 0:
                    t_inverse += inverse_scaler(enc_X[[0]], location[[0]], location_num, scaler).tolist()
                enc_X = enc_X.to(DEVICE)
                dec_X = dec_X.to(DEVICE)
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
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    train_dataloader, val_dataloader, test_dataloader, scaler, location2id = get_dataloader(
        args.X_seq, args.t_seq, True, args.mode
    )

    # 方法A: 訓練データ, 検証データ, テストデータ (EarlyStoppingに検証データを使用)
    epoch, train_loss_list, val_loss_list, train_mae_list, val_mae_list, net = run(
        train_dataloader, val_dataloader, 100000, args.patience, "Train And Val"
    )
    plot_history(train_loss_list, val_loss_list, train_mae_list, val_mae_list, "train_and_val")
    test_loss, test_mae = test(net, test_dataloader, scaler)
    tqdm.write(f"Test Loss: {test_loss:.3f} | Test mae {test_mae:.3f}")
    plot_predict(net, train_dataloader, location2id, scaler, args.mode + "/train", "train_and_val")
    plot_predict(net, test_dataloader, location2id, scaler, args.mode, "train_and_val")

    train_dataloader, _, test_dataloader, scaler, location2id = get_dataloader(args.X_seq, args.t_seq, False, args.mode)

    # 方法B: (訓練データ + 検証データ), テストデータ (EarlyStoppingなし、epochは方法Aを使用)
    _, train_loss_list, _, train_mae_list, _, net = run(train_dataloader, None, epoch, args.patience, "Train Only")
    plot_history(train_loss_list, None, train_mae_list, None, "train_only")
    test_loss, test_mae = test(net, test_dataloader, scaler)
    tqdm.write(f"Test Loss: {test_loss:.3f} | Test mae {test_mae:.3f}")
    plot_predict(net, train_dataloader, location2id, scaler, args.mode + "/train", "train_only")
    plot_predict(net, test_dataloader, location2id, scaler, args.mode, "train_only")

    # 方法C: (訓練データ + 検証データ), テストデータ (EarlyStoppingにテストデータを使用)
    _, train_loss_list, val_loss_list, train_mae_list, val_mae_list, net = run(
        train_dataloader, test_dataloader, 100000, args.patience, "Leak"
    )
    plot_history(train_loss_list, val_loss_list, train_mae_list, val_mae_list, "leak")
    test_loss, test_mae = test(net, test_dataloader, scaler)
    tqdm.write(f"Test Loss: {test_loss:.3f} | Test mae {test_mae:.3f}")
    plot_predict(net, train_dataloader, location2id, scaler, args.mode + "/train", "leak")
    plot_predict(net, test_dataloader, location2id, scaler, args.mode, "leak")

#       epoch | test loss | test mae
# ------------------------------------
# 方法A: 515   | 261.876   | 7004.342
# 方法A: 494   | 126.688   | 5610.677
# 方法C: 410  | 126.671   | 5569.648
