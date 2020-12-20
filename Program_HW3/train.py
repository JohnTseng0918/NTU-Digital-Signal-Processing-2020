import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from autoencoder import autoencoder
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default = 10, help="number of epoch", type=int)
parser.add_argument("--num", default = "0,1,2,3,4,5,6,7,8,9", help="number", type=str)
args = parser.parse_args()

epoch = args.epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = MNIST('./data', download=True, train=True, transform=transforms.ToTensor())

tar = list(map(int, args.num.split(",")))
trainset = get_target(trainset, tar)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

model = autoencoder()
model = model.to(device)

model.train()
optimizer = optim.Adam(model.parameters())

for i in range(epoch):
    total_loss = 0.0
    for data in trainloader:
        x, _ = data
        groud_truth = x.clone()

        x = transforms.RandomErasing(p=1,value=0.5,scale=(0.1, 0.3))(x)
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