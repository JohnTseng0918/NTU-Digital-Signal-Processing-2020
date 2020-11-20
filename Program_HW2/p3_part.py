import numpy as np
import pandas as pd
from PIL import Image
from omp import OMP

# label = k training data

num_list = [3]

# load training data and get label = k training data
base_data_frame = pd.read_csv("mnist_train.csv")
base_data_frame = base_data_frame[base_data_frame['label'].isin(num_list)]
np_base_data = base_data_frame.iloc[:,1:].to_numpy(dtype = float)
# load testing data and get label = k testing data
test_data_frame = pd.read_csv("mnist_test.csv")
test_data_frame = test_data_frame[test_data_frame['label'].isin(num_list)]
np_test_data = test_data_frame.iloc[:,1:].to_numpy()

# row: #data, col: #dim 
row, col = np_base_data.shape

idx = 87
preserve_list = [25, 50, 95]
ne_list = []

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
    ne_list.append(num_ev)
    return u[:,:num_ev]

img = Image.fromarray((np_test_data[idx]*1.0).reshape(28,28)).convert("RGB")
img.save("3_part_origin","JPEG")

mean_data = np.mean(np_base_data, axis = 0, keepdims = True)
np_data = np_base_data - mean_data

# center PCA
for x in preserve_list:
    p = PCA(np_data.T, x)
    np_data_compress = p @ p.T @ (np_test_data - mean_data).T
    np_data_compress = np_data_compress.T + mean_data

    loss = sum((np_test_data[idx]-np_data_compress[idx])**2)**0.5
    print("(3) idx:", idx,"center PCA, energy preserve:", x, "euclidean distance: ",loss)
    idx_data = np_data_compress[idx,:].reshape(28,28)
    img = Image.fromarray(idx_data).convert("RGB")
    s = "3_center_" + "energy_" + str(x) + "_k"
    img.save(s,"JPEG")

for sparse in ne_list:
    recover_signal , error = OMP(sparse, np_test_data[idx], np_base_data)
    img = Image.fromarray(recover_signal.reshape(28,28)).convert("RGB")
    s = "3_OMP_" + str(sparse) + "_k"
    img.save(s, "JPEG")
    print("(3) idx:", idx, "OMP, sparsity:", sparse, "euclidean distance:", error)