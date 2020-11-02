import numpy as np
import pandas as pd
from PIL import Image

data_frame = pd.read_csv("mnist_train.csv")
np_data = data_frame.iloc[:,1:].to_numpy()
mean_data = np.mean(np_data, axis = 0)
np_data = np_data - mean_data

def PCA(matrix, num_eigen): # Principle Component Analysis
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    return u[:,:num_eigen]

p = PCA(np_data.T, 3)

for i in range(3):
    tmp = p[:,i]
    tmp = tmp / np.max(tmp)
    tmp *= 255
    tmp = tmp.reshape(28,28)
    img = Image.fromarray(tmp).convert("RGB")
    s = "eigen_all_" + str(i)
    img.save(s,"JPEG")

for i in range(10):
    data_frame_part = data_frame[data_frame['label'].isin([i])]
    np_data = data_frame_part.iloc[:,1:].to_numpy()
    mean_data = np.mean(np_data, axis = 0)
    np_data = np_data - mean_data
    p = PCA(np_data.T, 3)

    for j in range(3):
        tmp = p[:,j]
        tmp = tmp / np.max(tmp)
        tmp *= 255
        tmp = tmp.reshape(28,28)
        img = Image.fromarray(tmp).convert("RGB")
        s = "eigen" + str(i) + "_" + str(j)
        img.save(s,"JPEG")