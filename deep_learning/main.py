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
    mae /= dataloader.dataset.size
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
    mae /= dataloader.dataset.size
    return loss, mae


def plot_history(train_loss_list, val_loss_list, train_mae_list, val_mae_list):
    plt.figure()
    plt.plot(train_loss_list, label="train")
    plt.plot(val_loss_list, label="val")
    plt.legend()
    plt.title("loss")
    plt.savefig("loss.png")
    plt.figure()
    plt.plot(train_mae_list, label="train")
    plt.plot(val_mae_list, label="val")
    plt.legend()
    plt.title("acc")
    plt.savefig("mae.png")


def plot_predict(net, dataloader, location2id, scaler, mode):
    # len(dataloader) == 1の時だけ
    for location_str, location_id in tqdm(location2id.items()):
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
        for i in range(0, len(y_inverse), 4):
            if i > 0:
                ax.plot(range(i - 1, i + 1), y_inverse[i - 1 : i + 1], color="C0", linestyle="--")
            ax.plot(range(i, i + 4), y_inverse[i : i + 4], color="C0")
        ax.lines[0].set_label("predict")
        ax.plot(t_inverse, label="ground truth", color="C1")
        ax.legend()
        ax.set_title(f"sequential_{location_str}_{mae:.1f}.png")
        fig.savefig(f"deep_learning/result/{mode}/sequential_{location_str}.png")
        plt.close(fig)

        # 相関係数
        # plt.figure()
        # r = np.corrcoef(t_inverse, y_inverse)[0][1]
        # plt.plot(t_inverse, y_inverse, "o")
        # plt.axis("square")
        # plt.xlabel("ground truth")
        # plt.ylabel("predict")
        # plt.title(f"r_{location_str}_{r:.3f}.png")
        # plt.savefig(f"deep_learning/result/r_{location_str}.png")
        # plt.close()


    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["World", "Japan", "Tokyo"], default="World")
    parser.add_argument("--X_seq", type=int, default=10)
    parser.add_argument("--t_seq", type=int, default=4)
    args = parser.parse_args()
net = transformer_net.TransformerNet(
    d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.2
).to(DEVICE)
    train_dataloader, val_dataloader, test_dataloader, scaler, location2id = get_dataloader(X_seq, t_seq, args.mode)

train_loss_list = []
val_loss_list = []
train_mae_list = []
val_mae_list = []
epoch = 10000
optimizer = optim.Adam(net.parameters(), lr=1e-5)
pbar = tqdm(total=epoch, position=0)
desc = tqdm(total=epoch, position=1, bar_format="{desc}", desc="")
for epoch in range(epoch):
    train_loss, train_mae = train(net, optimizer, train_dataloader, scaler)
    val_loss, val_mae = val_test(net, val_dataloader, scaler)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    train_mae_list.append(train_mae)
    val_mae_list.append(val_mae)
    if early_stopping(net, val_loss):
        break
    pbar.update(1)
    desc.set_description(
        f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Train mae: {train_mae:.3f} | Val mae: {val_mae:.3f} | Best Val Loss: {early_stopping.best_value:.3f} | EaryStopping Counter: {early_stopping.counter}/{early_stopping.patience}"
    )
    # tqdm.write(
    #     f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Train mae: {train_mae:.3f} | Val mae: {val_mae:.3f} | Best Val Loss: {early_stopping.best_value:.3f} | EaryStopping Counter: {early_stopping.counter}/{early_stopping.patience}"
    # )
pbar.clear()
desc.clear()
plot_history(train_loss_list, val_loss_list, train_mae_list, val_mae_list)

test_loss, test_mae = val_test(net, test_dataloader, scaler)
tqdm.write(f"Test Loss: {test_loss:.3f} | Test mae {test_mae:.3f}")
# print()
mae_list = []
# for i in range(47):
    mae_list.append(plot_predict(net, test_dataloader, location2id, scaler, args.mode))
