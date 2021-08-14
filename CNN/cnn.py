import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class LeNet(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, c_out)
        
    def forward(self, x):
        out = F.max_pool2d(F.relu(self.conv1(x)), 2)
        out = F.max_pool2d(F.relu(self.conv2(out)), 2)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class SimpleNet(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        channel = 32
        self.conv1 = nn.Conv2d(c_in, 2*channel, kernel_size=3)
        self.conv2 = nn.Conv2d(2*channel, 4*channel, kernel_size=3)
        self.fc1 = nn.Linear(4*channel*4*4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, c_out)

    def forward(self, x):
        out = F.max_pool2d(F.relu(self.conv1(x)), 2)
        out = F.max_pool2d(F.relu(self.conv2(out)), 2)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
