import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv1d(2, 4, kernel_size=7, stride=2, padding=3, groups=2)
        self.bn1 = nn.BatchNorm1d(4) 
        self.conv2 = nn.Conv1d(4,8,kernel_size=7, stride=2, padding=3, groups=4)
        self.bn2 = nn.BatchNorm1d(8) 
        self.conv3 = nn.Conv1d(8,16,kernel_size=7, stride=2, padding=3, groups=8)
        self.bn3 = nn.BatchNorm1d(16) 
        self.conv4 = nn.Conv1d(16,32,kernel_size=3, stride=2, padding=1, groups=8)
        self.bn4 = nn.BatchNorm1d(32) 
        self.conv5 = nn.Conv1d(32,64,kernel_size=3, stride=2, padding=1, groups=8)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64*625, 4)
    def forward(self, x):
        x = self.bn1(torch.sigmoid(self.conv1(x)))
        x = self.bn2(torch.tanh(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        x = x.view(-1, 64*625)
        y = self.fc(x)
        return y