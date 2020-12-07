import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default = 20, help="number of epoch", type=int)
parser.add_argument("--num", default = "0,1,2,3,4,5,6,7,8,9", help="number", type=str)
args = parser.parse_args()

epoch = args.epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = MNIST('./data', download=True, train=True, transform=transforms.ToTensor())
testset = MNIST('./data', download=True, train=False, transform=transforms.ToTensor())

def get_target(dataset, target_label):
    samples, targets = [], []
    for sample, target in zip(dataset.data, dataset.targets):
        if target in target_label:
            samples.append(sample)
            targets.append(target)
    dataset.data, dataset.targets = torch.stack(samples), torch.stack(targets)
    return dataset

tar = list(map(int, args.num.split(",")))
trainset, testset = get_target(trainset, tar), get_target(testset, tar)


trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128,shuffle=False)

class autoencoder(nn.Module):
    def __init__(self, hidden_dim = 128):
        super(autoencoder, self).__init__()
        self.encoder = nn.Linear(28*28, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 28*28)
    def forward(self, x):
        y = self.decoder(self.encoder(x))
        return y

model = autoencoder(hidden_dim=128)
model = model.to(device)

model.train()
optimizer = optim.Adam(model.parameters())

def random_inpainting(x):
    b, c, h, w = x.shape
    for i in range(b):
        inpainting_size_h = random.randint(1,6)
        inpainting_size_w = random.randint(1,6)
        nh = random.randint(0, h-inpainting_size_h)
        nw = random.randint(0, w-inpainting_size_w)
        for j in range(nh, nh + inpainting_size_h):
            for k in range(nw, nw + inpainting_size_w):
                x[i][0][j][k] = 0.5
    return x


for i in range(epoch):
    total_loss = 0.0
    for data in trainloader:
        x, _ = data
        groud_truth = x.clone()

        x = random_inpainting(x)
        x = x.to(device)
        groud_truth = groud_truth.to(device)
        
        prediction = model(x.view(x.size(0),-1))
        loss = F.mse_loss(prediction, groud_truth.view(groud_truth.size(0),-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print("epoch:", i+1, "total_loss:",total_loss)

torch.save(model.state_dict(), "autoencoder.pth")