import sys
sys.path.append('..')
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from MLP.mlp import MLP
from CNN.cnn import LeNet
from common.utils import plot_training_curve


# load data
data_path = '../dataset/MNIST'

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(),
    # transforms.RandomHorizontalFlip()
])
mnist = datasets.MNIST(data_path, train=True, transform=transform, download=False)
mnist_val = datasets.MNIST(data_path, train=False, transform=transform, download=False)
batch_size = 64
train_loader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(mnist_val, batch_size=batch_size, shuffle=True)


n_epochs = 15
learning_rate = 1e-2

model = MLP(784, 516, 10)
# model = LeNet(1, 10)
device = 'gpu' if torch.cuda.is_available() else 'cpu'
model.to(device=device)
# loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


train_epoch_loss = []
val_epoch_loss = []
start = time.time()
for epoch in range(1, n_epochs+1):
    train_loss = []
    val_loss = []
    model.train()
    for x, t in train_loader:
        n_batch = x.shape[0]
        # x = x.view(n_batch, -1).to(device)
        x = x.to(device)
        t = t.to(device)

        y_pred = model(x)
        loss = nn.CrossEntropyLoss()(y_pred, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.detach().numpy())


    model.eval()
    for x, t in val_loader:
        n_batch = x.shape[0]
        # x = x.view(n_batch, -1).to(device)
        x = x.to(device)
        t = t.to(device)

        y_pred = model(x)
        loss = nn.CrossEntropyLoss()(y_pred, t)

        val_loss.append(loss.detach().numpy())

    mean_train_loss = np.mean(train_loss)
    mean_val_loss = np.mean(val_loss)
    train_epoch_loss.append(mean_train_loss)
    val_epoch_loss.append(mean_val_loss)
    print('[{}] Epoch {}, Train loss {:.3f}, Val loss {:.3f}, Time {:.1f}[s]'.format(
        datetime.datetime.now(),
        epoch,
        mean_train_loss,
        mean_val_loss,
        time.time()-start
        ))

plot_training_curve(n_epochs, train_epoch_loss, val_epoch_loss)
