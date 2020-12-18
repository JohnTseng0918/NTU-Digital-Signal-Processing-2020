import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from autoencoder import autoencoder
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pic", default = 10, help="number of picture", type=int)
parser.add_argument("--num", default = "0,1,2,3,4,5,6,7,8,9", help="number", type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testset = MNIST('./data', download=True, train=False, transform=transforms.ToTensor())

tar = list(map(int, args.num.split(",")))
testset = get_target(testset, tar)

testloader = DataLoader(testset, batch_size=1, shuffle=True)

model = autoencoder()
model.load_state_dict(torch.load("./autoencoder.pth"))
model = model.to(device)
model.eval()

counter = 0
for data in testloader:
    x, _ = data
    groud_truth = x.clone()
    groud_truth_img = transforms.ToPILImage()(torch.squeeze(groud_truth))
    groud_truth_img.save("./pic/ground_"+str(counter+1),"JPEG")
    x = random_inpainting(x, 3)
    x_img = transforms.ToPILImage()(torch.squeeze(x))
    x_img.save("./pic/damage_"+str(counter+1),"JPEG")
    x = x.to(device)
    groud_truth = groud_truth.to(device)
        
    prediction = model(x.view(x.size(0),-1))
    loss = F.mse_loss(prediction, groud_truth.view(groud_truth.size(0),-1))
    prediction = prediction.to("cpu")
    prediction_img = torch.squeeze(prediction)
    prediction_img = transforms.ToPILImage()(prediction.view(1,28,28))
    prediction_img.save("./pic/reconstruction_"+str(counter+1),"JPEG")
    print(loss.item())
    counter += 1
    if counter == args.pic:
        break