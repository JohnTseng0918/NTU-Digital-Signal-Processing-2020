import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv1d(2, 4, kernel_size=7, stride=2, padding=3, groups=2)
        self.bn1 = nn.BatchNorm1d(4) 
        self.conv2 = nn.Conv1d(4,8,kernel_size=7, stride=2, padding=3, groups=2)
        self.bn2 = nn.BatchNorm1d(8) 
        self.conv3 = nn.Conv1d(8,16,kernel_size=7, stride=2, padding=3, groups=2)
        self.bn3 = nn.BatchNorm1d(16) 
        self.conv4 = nn.Conv1d(16,32,kernel_size=7, stride=2, padding=3, groups=2)
        self.bn4 = nn.BatchNorm1d(32) 
        self.fc = nn.Linear(32*1250, 4)
    def forward(self, x):
        x = self.bn1(torch.sigmoid(self.conv1(x)))
        x = self.bn2(torch.tanh(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = x.view(-1, 32*1250)
        y = self.fc(x)
        return y