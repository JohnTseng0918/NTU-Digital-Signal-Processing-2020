import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model import model

test_data = np.load("./Data/testdata.npy")

class Test_Dataset(Dataset):
    def __init__(self, x):
        self.data = torch.from_numpy(x)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
testset = Test_Dataset(test_data)
test_loader = DataLoader(dataset = testset, batch_size = 1, shuffle= False)

model = model()
model.load_state_dict(torch.load("./model.pth"))
model = model.to(device)

def inference(test_loader, device, model, ans_list):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            predict = model(data)
            ans = torch.argmax(predict, dim=1)
            ans_list.append(ans.item())

ans_list = []
inference(test_loader, device,model,ans_list)

df = pd.DataFrame({'id':pd.Series(list(range(len(ans_list)))),'category':pd.Series(ans_list)})
df.to_csv("./myAns.csv",index=False)