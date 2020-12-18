import torch
import random

def get_target(dataset, target_label):
    samples, targets = [], []
    for sample, target in zip(dataset.data, dataset.targets):
        if target in target_label:
            samples.append(sample)
            targets.append(target)
    dataset.data, dataset.targets = torch.stack(samples), torch.stack(targets)
    return dataset

def random_inpainting(x, num_inpaint):
    b, c, h, w = x.shape
    for i in range(b):
        # number of inpainting
        num = random.choice(range(1, num_inpaint + 1))
        for n in range(num):
            inpainting_size_h = random.randint(6, 12)
            inpainting_size_w = random.randint(6, 12)
            nh = random.randint(0, h-inpainting_size_h)
            nw = random.randint(0, w-inpainting_size_w)
            for j in range(nh, nh + inpainting_size_h):
                for k in range(nw, nw + inpainting_size_w):
                    x[i][0][j][k] = 0.5
    return x