import pandas as pd
import numpy as np
from PIL import Image

data_frame = pd.read_csv("mnist_train.csv")
np_data = data_frame.iloc[:,1:].to_numpy()
mean_data = np.mean(np_data, axis = 0).reshape(28,28)
img = Image.fromarray(mean_data).convert("RGB")
img.save("average","JPEG")

for i in range(10):
    data_frame_part = data_frame[data_frame['label'].isin([i])]
    np_data = data_frame_part.iloc[:,1:].to_numpy()
    mean_data = np.mean(np_data, axis = 0).reshape(28,28)
    img = Image.fromarray(mean_data).convert("RGB")
    s = "average" + str(i)
    img.save(s,"JPEG")