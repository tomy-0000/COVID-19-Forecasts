#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

import transformer_net
from utils import EarlyStopping, get_dataloader

sns.set()


def train(net, optimizer, dataloader, min_max_scaler):
    net.train()
    loss = 0.0
    mae = 0.0
    for X, t, prefecture in tqdm(dataloader, total=len(dataloader), leave=False, position=2):
        X = X.cuda()
        t = t.cuda()
        optimizer.zero_grad()
        y = net(X, t)
        batch_loss = F.mse_loss(y, t)
        batch_loss.backward()
        optimizer.step()
        loss += F.mse_loss(y, t, reduction="sum").cpu().detach().item()
        batch_mae = F.l1_loss(y, t, reduction="none").cpu().detach().numpy()
        batch_mae = batch_mae.reshape(-1)
        location_num = dataloader.dataset.location_num
        placeholder = np.zeros([len(batch_mae), location_num])
        prefecture = prefecture.numpy().repeat(t.shape[-1])
        placeholder[range(len(placeholder)), prefecture] = batch_mae.reshape(-1)
        mae += min_max_scaler.inverse_transform(placeholder)[range(len(placeholder)), prefecture].sum()
    loss /= dataloader.dataset.size
    mae /= dataloader.dataset.size
    return loss, mae


def val_test(net, dataloader, min_max_scaler):
    net.eval()
    loss = 0.0
    mae = 0.0
    for X, t, prefecture in tqdm(dataloader, total=len(dataloader), leave=False, position=2):
        X = X.cuda()
        t = t.cuda()
        y = net(X, t)
        loss = F.mse_loss(y, t, reduction="sum").item()
        batch_mae = F.l1_loss(y, t, reduction="none").cpu().detach().numpy()
        batch_mae = batch_mae.reshape(-1)
        location_num = dataloader.dataset.location_num
        placeholder = np.zeros([len(batch_mae), location_num])
        prefecture = prefecture.numpy().repeat(t.shape[-1])
        placeholder[range(len(placeholder)), prefecture] = batch_mae.reshape(-1)
        mae += min_max_scaler.inverse_transform(placeholder)[range(len(placeholder)), prefecture].sum()
    loss /= dataloader.dataset.size
    mae /= dataloader.dataset.size
    return loss, mae


net = transformer_net.TransformerNet().cuda()
train_dataloader, val_dataloader, test_dataloader, min_max_scaler = get_dataloader(10, 4, "Tokyo")
early_stopping = EarlyStopping(10)

train_loss_list = []
val_loss_list = []
train_mae_list = []
val_mae_list = []
optimizer = optim.AdamW(net.parameters(), lr=1e-5)
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
        pbar.clear()
        desc.clear()
        break
    pbar.update(1)
    desc.set_description(
        f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Train mae: {train_mae:.3f} | Val mae: {val_mae:.3f} | Best Val Loss: {early_stopping.best_value:.3f} | EaryStopping Counter: {early_stopping.counter}/{early_stopping.patience}"
    )
    tqdm.write(
        f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Train mae: {train_mae:.3f} | Val mae: {val_mae:.3f} | Best Val Loss: {early_stopping.best_value:.3f} | EaryStopping Counter: {early_stopping.counter}/{early_stopping.patience}"
    )
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
test_loss, test_mae = val_test(net, test_dataloader, min_max_scaler)
tqdm.write(f"Test Loss: {test_loss:.3f} | Test mae {test_mae:.3f}")
