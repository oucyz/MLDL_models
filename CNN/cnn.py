import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class LeNet(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, c_out)
        
    def forward(self, x):
        out = F.max_pool2d(F.relu(self.conv1(x)))
        out = F.max_pool2d(F.relu(self.conv2(out)))
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
