import numpy as np
import pandas as pd
from PIL import Image
from omp import OMP

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

# normalize
for i in range(row):
    norm = np.linalg.norm(np_base_data[i])
    np_base_data[i] /= norm

# problem 2
total_error = 0.0
for i in range(30):
    recover_signal , error = OMP(10, np_test_data[i], np_base_data)
    total_error += error
print("(2) sparsity: 10, 30 testing data Euclidean distance sum: ", total_error)

# output image
idx = 87

img = Image.fromarray((np_test_data[idx]*1.0).reshape(28,28)).convert("RGB")
img.save("2_origin","JPEG")

recover_signal , error = OMP(10, np_test_data[idx], np_base_data)
img = Image.fromarray(recover_signal.reshape(28,28)).convert("RGB")
img.save("2_10","JPEG")
print("(2) sparsity: 10, idx:", idx, "error:", error)