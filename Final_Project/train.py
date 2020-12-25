import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from model import *
from utils import *

train_data, train_label = np.load("./Data/traindata.npy"), np.load("./Data/trainlabel.npy")
val_data, val_label = np.load("./Data/validationdata.npy"), np.load("./Data/validationlabel.npy")

class Bearing_Health_Condition_Dataset(Dataset):
    def __init__(self, x, y):
        self.data = torch.from_numpy(x)
        self.label = torch.from_numpy(y)
        self.label = self.label.type(torch.LongTensor)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainset = Bearing_Health_Condition_Dataset(train_data, train_label)
validationset = Bearing_Health_Condition_Dataset(val_data, val_label)
train_loader = DataLoader(dataset = trainset, batch_size = 128, shuffle= True)
val_loader = DataLoader(dataset = validationset, batch_size = 64, shuffle=False)
model = model().to(device)

num_epoch = 50
optimizer = optim.Adam(model.parameters(),weight_decay=0.001)

def evaluation(outputs, labels):
    correct = torch.sum(torch.eq(torch.argmax(outputs, dim=1), labels)).item()
    return correct

def train(train_loader, device, model, optimizer, criterion, epoch, num_data):
    model.train()
    total_acc = 0
    total_loss = 0.0
    num_batch = len(train_loader)
    for i, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        predict = model(data)
        loss = criterion(predict, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = evaluation(predict, label)
        total_acc += correct
        total_loss += loss.item()
        print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(epoch+1, i+1, num_batch, loss.item(), correct*100/len(data)), end='\r')
    print('\nTrain | Average Loss:{:.5f} Total Acc: {:.3f}'.format(total_loss/num_batch, total_acc*100/num_data))


def validation(val_loader, device, model, criterion, num_data):
    model.eval()
    total_acc = 0
    total_loss = 0.0
    num_batch = len(val_loader)
    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            data = data.to(device)
            label = label.to(device)
            predict = model(data)
            loss = criterion(predict, label)

            correct = evaluation(predict, label)
            total_acc += correct
            total_loss += loss.item()
        print("Valid | Average Loss:{:.5f} Total Acc: {:.3f} ".format(total_loss/num_batch, total_acc*100/num_data))
    
    return total_acc



criterion = nn.CrossEntropyLoss()
maxacc = 0
for epoch in range(num_epoch):
    train(train_loader, device, model, optimizer, criterion, epoch, len(train_data))
    acc = validation(val_loader,device,model,criterion,len(val_data))
    if maxacc < acc:
        maxacc = acc
        torch.save(model.state_dict(), "model.pth")
        print("Get the Highest Accuracy, Save the model")
    print("--------------------------------------------------------")