import torch

def get_target(dataset, target_label):
    samples, targets = [], []
    for sample, target in zip(dataset.data, dataset.targets):
        if target in target_label:
            samples.append(sample)
            targets.append(target)
    dataset.data, dataset.targets = torch.stack(samples), torch.stack(targets)
    return dataset