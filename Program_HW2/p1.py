import numpy as np
import pandas as pd
from PIL import Image
from omp import OMP

# load training data
base_data_frame = pd.read_csv("mnist_train.csv")
np_base_data = base_data_frame.iloc[:,1:].to_numpy(dtype = float)
# load testing data
test_data_frame = pd.read_csv("mnist_test.csv")
np_test_data = test_data_frame.iloc[:,1:].to_numpy()

# row: #data, col: #dim
row, col = np_base_data.shape

# normalize
for i in range(row):
    norm = np.linalg.norm(np_base_data[i])
    np_base_data[i] /= norm


# 1.a
total_error = 0.0
for i in range(30):
    recover_signal , error = OMP(5, np_test_data[i], np_base_data)
    total_error += error
print("(1.a) sparsity: 5, 30 testing data Euclidean distance sum: ", total_error)

total_error = 0.0
for i in range(30):
    recover_signal , error = OMP(10, np_test_data[i], np_base_data)
    total_error += error
print("(1.a) sparsity: 10, 30 testing data Euclidean distance sum: ", total_error)

total_error = 0.0
for i in range(30):
    recover_signal , error = OMP(150, np_test_data[i], np_base_data)
    total_error += error
print("(1.a) sparsity: 150, 30 testing data Euclidean distance sum: ", total_error)

# 1.a output image and loss
idx = 87

img = Image.fromarray((np_test_data[idx]*1.0).reshape(28,28)).convert("RGB")
img.save("1a_origin","JPEG")

recover_signal , error = OMP(5, np_test_data[idx], np_base_data)
img = Image.fromarray(recover_signal.reshape(28,28)).convert("RGB")
img.save("1a5","JPEG")
print("(1.a) sparsity: 5, idx:", idx, "error:", error)

recover_signal , error = OMP(10, np_test_data[idx], np_base_data)
img = Image.fromarray(recover_signal.reshape(28,28)).convert("RGB")
img.save("1a10","JPEG")
print("(1.a) sparsity: 10, idx:", idx, "error:", error)

recover_signal , error = OMP(150, np_test_data[idx], np_base_data)
img = Image.fromarray(recover_signal.reshape(28,28)).convert("RGB")
img.save("1a150","JPEG")
print("(1.a) sparsity: 150, idx:", idx, "error:", error)


# 1.b base 0~9999 output image and loss

total_error = 0.0
for i in range(30):
    recover_signal , error = OMP(5, np_test_data[i], np_base_data[:10000])
    total_error += error
print("(1.b) sparsity: 5, 30 testing data Euclidean distance sum: ", total_error)

total_error = 0.0
for i in range(30):
    recover_signal , error = OMP(10, np_test_data[i], np_base_data[:10000])
    total_error += error
print("(1.b) sparsity: 10, 30 testing data Euclidean distance sum: ", total_error)

total_error = 0.0
for i in range(30):
    recover_signal , error = OMP(150, np_test_data[i], np_base_data[:10000])
    total_error += error
print("(1.b) sparsity: 150, 30 testing data Euclidean distance sum: ", total_error)

idx = 87

img = Image.fromarray((np_test_data[idx]*1.0).reshape(28,28)).convert("RGB")
img.save("1b_origin","JPEG")

recover_signal , error = OMP(5, np_test_data[idx], np_base_data[:10000])
img = Image.fromarray(recover_signal.reshape(28,28)).convert("RGB")
img.save("1b5","JPEG")
print("(1.b) sparsity: 5, idx:", idx, "error:", error)

recover_signal , error = OMP(10, np_test_data[idx], np_base_data[:10000])
img = Image.fromarray(recover_signal.reshape(28,28)).convert("RGB")
img.save("1b10","JPEG")
print("(1.b) sparsity: 10, idx:", idx, "error:", error)

recover_signal , error = OMP(150, np_test_data[idx], np_base_data[:10000])
img = Image.fromarray(recover_signal.reshape(28,28)).convert("RGB")
img.save("1b150","JPEG")
print("(1.b) sparsity: 150, idx:", idx, "error:", error)