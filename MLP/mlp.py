import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()

        self.linear1 = nn.Linear(n_in, n_hidden)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_out)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.relu1(self.linear1(x))
        out = self.relu2(self.linear2(out))
        return out
