import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt

class KaggleNet(nn.Module):
    
    
    def __init__(self):
        super().__init__()

        #Convolutional
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.2)
        
        #Fully connected
        self.fc1 = nn.Linear(64*7*7, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 31)

    def forward(self, xin):
        # x is [batch_size, channels, heigth, width] = [bs, 1, 28, 28]
        x = F.relu(self.bn1(self.conv1(xin)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        # x is [bs, 32, 14, 14]
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x,2) 
        x = self.dropout1(x)
       
       # x is [bs, 64, 7, 7]
        x = x.view(-1, 64*7*7 ) # flatten
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x