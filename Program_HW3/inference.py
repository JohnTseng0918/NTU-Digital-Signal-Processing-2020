import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from autoencoder import autoencoder
from utils import *
import argparse

# read arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pic", default = 10, help="number of picture", type=int)
parser.add_argument("--num", default = "0,1,2,3,4,5,6,7,8,9", help="number", type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# process dataset
testset = MNIST('./data', download=True, train=False, transform=transforms.ToTensor())
tar = list(map(int, args.num.split(",")))
testset = get_target(testset, tar)

testloader = DataLoader(testset, batch_size=1, shuffle=True)

# declare model and load model
model = autoencoder()
model.load_state_dict(torch.load("./autoencoder.pth"))
model = model.to(device)
model.eval()

counter = 0
for data in testloader:
    x, _ = data

    # save origin image as ground truth
    groud_truth = x.clone()
    groud_truth_img = transforms.ToPILImage()(torch.squeeze(groud_truth))
    groud_truth_img.save("./pic/ground_"+str(counter+1),"JPEG")

    # RandomErasing image
    x = transforms.RandomErasing(p=1,value=0.5,scale=(0.1, 0.3))(x)
    x_img = transforms.ToPILImage()(torch.squeeze(x))
    x_img.save("./pic/damage_"+str(counter+1),"JPEG")
    
    # to device (GPU)
    x = x.to(device)
    groud_truth = groud_truth.to(device)    
        
    prediction = model(x.view(x.size(0),-1))
    loss = F.mse_loss(prediction, groud_truth.view(groud_truth.size(0),-1))
    
    # process prediction image
    prediction = prediction.to("cpu")
    # normalize
    prediction -= torch.min(prediction)
    prediction /= torch.max(prediction)
    # postprocesing
    prediction = postprocessing(groud_truth, x, prediction)
    prediction_img = torch.squeeze(prediction)
    prediction_img = transforms.ToPILImage()(prediction.view(1,28,28))
    prediction_img.save("./pic/reconstruction_"+str(counter+1),"JPEG")

    loss_2 = F.mse_loss(prediction, groud_truth.view(groud_truth.size(0),-1).to("cpu"))
    
    print(loss.item(),loss_2.item())
    counter += 1
    if counter == args.pic:
        break