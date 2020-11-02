import numpy as np
import pandas as pd
from PIL import Image
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--center", default = 1, help="center PCA (1) or non-center PCA (0)", type=int, choices=[0,1])
parser.add_argument("--num", default = "0,1,2,3,4,5,6,7,8,9", help="number", type=str)
args = parser.parse_args()

if args.center==1:
    print("cener PCA, number range: ",args.num)
else:
    print("non-center PCA, number range: ",args.num)

# set numbers
num_list = list(map(int, args.num.split(",")))

# read data and put them into padas data frame
data_frame = pd.read_csv("mnist_train.csv")
data_frame = data_frame[data_frame['label'].isin(num_list)]

# transform dataframe number to np array
np_data = data_frame.iloc[:,1:].to_numpy()
num_data, dim = np_data.shape

idx_list = random.sample(range(num_data),k=2)
#idx_list = []
print(idx_list)
preserve_list = [25,50,95]

def PCA(matrix, energy_threshold): # Principle Component Analysis
    # SVD
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    # singular value to eigenvalue
    ev = s**2
    
    # calculate eigenvalue energy preserve percentage
    num_ev = len(ev)
    for i in range(len(ev)):
        energy = sum(ev[:i+1]) * 100 / sum(ev)
        if energy >= energy_threshold:
            num_ev = i+1
            break
    return u[:,:num_ev]

if args.center == 1:
    mean_data = np.mean(np_data, axis = 0, keepdims = True)
    np_data = np_data - mean_data

for x in preserve_list:
    p = PCA(np_data.T, x)
    np_data_compress = p @ p.T @ np_data.T

    if args.center == 1:
        np_data_compress = np_data_compress.T + mean_data
    else:
        np_data_compress = np_data_compress.T

    for idx in idx_list:
        idx_data = np_data_compress[idx,:].reshape(28,28)
        img = Image.fromarray(idx_data).convert("RGB")
        if args.center==1:
            s = "center_" + "idx_" + str(idx) + "_energy_" + str(x)
        else:
            s = "noncenter_" + "idx_" + str(idx) + "_energy_" + str(x)
        img.save(s,"JPEG")