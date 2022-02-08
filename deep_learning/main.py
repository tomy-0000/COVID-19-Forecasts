import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

import transformer_net
from utils import EarlyStopping, get_dataloader, DEVICE

sns.set()
torch.manual_seed(111)
torch.backends.cudnn.benchmark = False


def inverse_scaler(x, location, location_num, scaler):
    x = x.detach().cpu().numpy()  # [N, T]
    location = location.numpy().repeat(x.shape[-1])  # [N*T]
    x = x.reshape(-1)  # [N*T]
    placeholder = np.zeros([len(x), location_num])  # [N*T, location_num]
    placeholder[range(len(placeholder)), location] = x
    x_inverse = scaler.inverse_transform(placeholder)[range(len(placeholder)), location]  # [N*T]
    return x_inverse


def train(net, optimizer, dataloader, scaler):
    net.train()
    loss = 0.0
    mae = 0.0
    for X, t, location in dataloader:
        X = X.to(DEVICE)
        t = t.to(DEVICE)
        optimizer.zero_grad()
        y = net(X, t)
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


def val_test(net, dataloader, scaler):
    net.eval()
    loss = 0.0
    mae = 0.0
    with torch.no_grad():
        for X, t, location in dataloader:
            X = X.to(DEVICE)
            t = t.to(DEVICE)
            y = net(X, t)
            loss += F.mse_loss(y, t, reduction="sum").detach().item()
            location_num = dataloader.dataset.location_num
            y_inverse = inverse_scaler(y, location, location_num, scaler)
            t_inverse = inverse_scaler(t, location, location_num, scaler)
            mae += abs(y_inverse - t_inverse).sum()
    loss = (loss / dataloader.dataset.size) ** 0.5
    mae = mae / dataloader.dataset.size
    return loss, mae


def run(train_dataloader, total_epoch, patience, val_dataloader=None):
    train_loss_list = []
    train_mae_list = []
    val_loss_list = []
    val_mae_list = []
    early_stopping = EarlyStopping(patience)
    if val_dataloader is None:
        tqdm.write("[Train Only]")
    else:
        tqdm.write("[Train And Val]")
    net = transformer_net.TransformerNet(
        d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.2
    ).to(DEVICE)
    optimizer = optim.AdamW(net.parameters(), lr=1e-5)
    pbar = tqdm(total=total_epoch, position=0)
    desc = tqdm(total=total_epoch, position=1, bar_format="{desc}", desc="")
    for epoch in range(total_epoch):
        train_loss, train_mae = train(net, optimizer, train_dataloader, scaler)
        train_loss_list.append(train_loss)
        train_mae_list.append(train_mae)
        if val_dataloader is not None:
            val_loss, val_mae = val_test(net, val_dataloader, scaler)
            val_loss_list.append(val_loss)
            val_mae_list.append(val_mae)
            if early_stopping(net, val_loss):
                break
        pbar.update(1)
        if val_dataloader is not None:
            desc_str = f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Train MAE: {train_mae:.3f} | Val MAE: {val_mae:.3f} | Best Val Loss: {early_stopping.best_value:.3f} | EaryStopping Counter: {early_stopping.counter}/{early_stopping.patience}"
        else:
            desc_str = f"Train Loss: {train_loss:.3f} | Train MAE: {train_mae:.3f}"
        desc.set_description(desc_str)
    epoch -= patience
    return epoch, train_loss_list, val_loss_list, train_mae_list, val_mae_list, net


def plot_history(train_loss_list, train_mae_list, suffix, val_loss_list=None, val_mae_list=None):
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
    _, t, _ = dataloader.dataset[0]
    t_seq = len(t)
    for location_str, location_id in tqdm(location2id.items(), leave=False):
        net.eval()
        y_inverse = []
        t_inverse = []
        cnt = 0
        with torch.no_grad():
            for X, t, location in dataloader:
                X = X[location == location_id]
                t = t[location == location_id]
                location = location[location == location_id]
                X = X.to(DEVICE)
                t = t.to(DEVICE)
                y = net(X, t)
                location_num = dataloader.dataset.location_num
                y_inverse += inverse_scaler(y, location, location_num, scaler).tolist()
                t_inverse += inverse_scaler(t, location, location_num, scaler).tolist()
                cnt += len(y_inverse)
        mae = abs(np.array(y_inverse) - np.array(t_inverse)).sum()
        mae /= cnt
        fig, ax = plt.subplots()
        for i in range(0, len(y_inverse), t_seq):
            if i > 0:
                ax.plot(range(i - 1, i + 1), y_inverse[i - 1 : i + 1], color="C0", linestyle="--")
            ax.plot(range(i, i + t_seq), y_inverse[i : i + t_seq], color="C0")
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

    train_dataloader, val_dataloader, train2_dataloader, test_dataloader, scaler, location2id = get_dataloader(
        args.X_seq, args.t_seq, args.mode
    )

    epoch, train_loss_list, val_loss_list, train_mae_list, val_mae_list, net = run(
        train_dataloader, 100000, args.patience, val_dataloader=val_dataloader
    )
    plot_history(
        train_loss_list, train_mae_list, "train_and_val", val_loss_list=val_loss_list, val_mae_list=val_mae_list
    )
    test_loss, test_mae = val_test(net, test_dataloader, scaler)
    tqdm.write(f"Test Loss: {test_loss:.3f} | Test mae {test_mae:.3f}")
    plot_predict(net, train_dataloader, location2id, scaler, args.mode + "/train", "")
    plot_predict(net, test_dataloader, location2id, scaler, args.mode, "train_and_val")

    epoch, train_loss_list, val_loss_list, train_mae_list, val_mae_list, net = run(
        train_dataloader, epoch, args.patience
    )
    plot_history(train_loss_list, train_mae_list, "train_only")
    test_loss, test_mae = val_test(net, test_dataloader, scaler)
    tqdm.write(f"Test Loss: {test_loss:.3f} | Test mae {test_mae:.3f}")
    plot_predict(net, test_dataloader, location2id, scaler, args.mode, "train_only")
