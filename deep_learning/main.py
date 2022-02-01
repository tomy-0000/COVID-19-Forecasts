#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

import transformer_net
from utils import EarlyStopping, get_dataloader

sns.set()


def inverse_min_max_scaler(x, location, location_num, min_max_scaler):
    x = x.detach().cpu().numpy()
    location = location.numpy().repeat(x.shape[-1])
    x = x.reshape(-1)
    placeholder = np.zeros([len(x), location_num])
    placeholder[range(len(placeholder)), location] = x
    x_inverse = min_max_scaler.inverse_transform(placeholder)[range(len(placeholder)), location]
    return x_inverse


def train(net, optimizer, dataloader, min_max_scaler):
    net.train()
    loss = 0.0
    mae = 0.0
    for X, t, location in dataloader:
        X = X.cuda()
        t = t.cuda()
        optimizer.zero_grad()
        y = net(X, t)
        print()
        batch_loss = torch.sqrt(F.mse_loss(y, t))
        batch_loss.backward()
        optimizer.step()
        loss += F.mse_loss(y, t, reduction="sum").detach().item() ** 0.5
        location_num = dataloader.dataset.location_num
        y_inverse = inverse_min_max_scaler(y, location, location_num, min_max_scaler)
        t_inverse = inverse_min_max_scaler(t, location, location_num, min_max_scaler)
        mae += abs(y_inverse - t_inverse).sum()
    loss /= dataloader.dataset.size
    mae /= dataloader.dataset.size
    return loss, mae


def val_test(net, dataloader, min_max_scaler):
    net.eval()
    loss = 0.0
    mae = 0.0
    with torch.no_grad():
        for X, t, location in dataloader:
            X = X.cuda()
            t = t.cuda()
            y = net(X, t)
            loss += F.mse_loss(y, t, reduction="sum").detach().item() ** 0.5
            location_num = dataloader.dataset.location_num
            y_inverse = inverse_min_max_scaler(y, location, location_num, min_max_scaler)
            t_inverse = inverse_min_max_scaler(t, location, location_num, min_max_scaler)
            mae += abs(y_inverse - t_inverse).sum()
    loss /= dataloader.dataset.size
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


def plot_predict(net, dataloader, location2id, min_max_scaler):
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
                X = X.cuda()
                t = t.cuda()
                y = net(X, t)
                location_num = dataloader.dataset.location_num
                y_inverse += inverse_min_max_scaler(y, location, location_num, min_max_scaler).tolist()
                t_inverse += inverse_min_max_scaler(t, location, location_num, min_max_scaler).tolist()
                cnt += len(y_inverse)
        mae = abs(np.array(y_inverse) - np.array(t_inverse)).sum()
        mae /= cnt
        plt.figure()
        plt.plot(y_inverse, label="predict")
        plt.plot(t_inverse, label="ground truth")
        plt.legend()
        plt.title(f"sequential_{location_str}_{mae:.1f}.png")
        plt.savefig(f"deep_learning/result/sequential_{location_str}.png")
        plt.close()
        # 相関係数
        r = np.corrcoef(t_inverse, y_inverse)[0][1]
        plt.plot(t_inverse, y_inverse, ".")
        plt.xlabel("ground truth")
        plt.xlabel("predict")
        plt.title(f"r_{location_str}_{r:.3f}.png")
        plt.savefig(f"deep_learning/result/r_{location_str}.png")
        plt.close()


net = transformer_net.TransformerNet(
    d_model=256, nhead=4, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.2
).cuda()
train_dataloader, val_dataloader, test_dataloader, min_max_scaler, location2id = get_dataloader(10, 4, "World")
early_stopping = EarlyStopping(20)

train_loss_list = []
val_loss_list = []
train_mae_list = []
val_mae_list = []
optimizer = optim.Adam(net.parameters(), lr=1e-5)
pbar = tqdm(total=1000, position=0)
desc = tqdm(total=1000, position=1, bar_format="{desc}", desc="")
for epoch in range(1000):
    train_loss, train_mae = train(net, optimizer, train_dataloader, min_max_scaler)
    val_loss, val_mae = val_test(net, val_dataloader, min_max_scaler)
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

test_loss, test_mae = val_test(net, test_dataloader, min_max_scaler)
tqdm.write(f"Test Loss: {test_loss:.3f} | Test mae {test_mae:.3f}")
# print()
mae_list = []
# for i in range(47):
mae_list.append(plot_predict(net, test_dataloader, location2id, min_max_scaler))
# print(sum(mae_list) / len(mae_list))
