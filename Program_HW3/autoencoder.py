import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self, hidden_dim):
        super(encoder, self).__init__()
        self.layer1 = nn.Linear(28*28, 512)
        self.layer2 = nn.Linear(512, hidden_dim)
    def forward(self, x):
        x = self.layer1(x)
        y = self.layer2(x)
        return y

class decoder(nn.Module):
    def __init__(self, hidden_dim):
        super(decoder, self).__init__()
        self.layer1 = nn.Linear(hidden_dim, 512)
        self.layer2 = nn.Linear(512, 28*28)
    def forward(self, x):
        x = self.layer1(x)
        y = self.layer2(x)
        return y

class autoencoder(nn.Module):
    def __init__(self, hidden_dim = 128):
        super(autoencoder, self).__init__()
        self.encoder = encoder(hidden_dim)
        self.decoder = decoder(hidden_dim)
    def forward(self, x):
        y = self.decoder(self.encoder(x))
        return y