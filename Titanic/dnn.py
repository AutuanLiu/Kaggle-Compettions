"""
Email: autuanliu@163.com
Date: 2018/10/30
"""

from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class MLP(nn.Module):
    def __init__(self, in_dim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_dim, 12)
        self.l2 = nn.Linear(12, 6)
        self.l3 = nn.Linear(6, 1)
        self.ac1 = nn.ReLU()
        self.ac2 = nn.Tanh()
        self.ac3 = nn.Sigmoid()
    
    def forward(self, x):
        y = self.ac1(self.l1(x))
        y = self.ac2(self.l2(y))
        y = self.ac3(self.l3(y))
        return y

class Titanic(Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.data = torch.from_numpy(data)
        self.target = torch.from_numpy(target)
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
    def __len__(self):
        return self.data.size(0)