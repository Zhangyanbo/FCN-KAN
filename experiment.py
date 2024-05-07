import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from kan_layer import KANLayer
from tqdm import tqdm


def target_fn(input):
    # f(x,y)=exp(sin(pi * x)+y^2)
    if len(input.shape) == 1:
        x, y = input
    else:
        x, y = input[:, 0], input[:, 1]
    return torch.exp(torch.sin(torch.pi * x) + y**2)


def train(model, n_epoch, lr=1e-2, batch_size=32, weight_decay=1e-3, eta_min=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epoch, eta_min=eta_min)
    lf = nn.MSELoss()

    loss_count = []

    for i in tqdm(range(n_epoch)):
        optimizer.zero_grad()
        x = torch.rand(batch_size, 2) * 2 - 1
        y = target_fn(x)
        y_pred = model(x)
        loss = lf(y_pred.reshape(-1), y)
        loss.backward()
        loss_count.append(loss.item())
        optimizer.step()
        scheduler.step()
    
    return loss_count


if __name__ == '__main__':
    model = nn.Sequential(
        KANLayer(2, 5),
        KANLayer(5, 1)
    )

    loss_count = train(model, 4000)

    plt.plot(loss_count)
    plt.semilogy()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    plt.close()

    layer = 0

    plt.figure(figsize=(10, 3))
    for i in range(2):
        for j in range(5):
            plt.subplot(2, 5, i*5+j+1)
            f = model[layer].take_function(i, j)

            x = torch.linspace(-1, 1, 100)
            y = f(x.unsqueeze(-1)).detach().squeeze()

            vmin, vmax = y.min().item() - 0.1, y.max().item() + 0.1

            plt.plot(x, y)
            plt.ylim(vmin, vmax)
            plt.title(f"$f_{{{i}, {j}}}$")

    plt.tight_layout()
    plt.savefig('layer_0.png')
    plt.close()

    layer = 1

    plt.figure(figsize=(10, 2))
    for i in range(5):
        for j in range(1):
            plt.subplot(1, 5, i*1+j*5+1)
            f = model[layer].take_function(i, j)

            x = torch.linspace(-1, 2, 100)
            y = f(x.unsqueeze(-1)).detach().squeeze()

            vmin, vmax = y.min().item() - 0.1, y.max().item() + 0.1

            plt.plot(x, y)
            plt.ylim(vmin, vmax)
            plt.title(f"$f_{{{i}, {j}}}$")

    plt.tight_layout()
    plt.savefig('layer_1.png')
    plt.close()