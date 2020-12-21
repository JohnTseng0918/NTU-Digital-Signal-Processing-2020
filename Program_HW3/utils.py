import torch

def get_target(dataset, target_label):
    samples, targets = [], []
    for sample, target in zip(dataset.data, dataset.targets):
        if target in target_label:
            samples.append(sample)
            targets.append(target)
    dataset.data, dataset.targets = torch.stack(samples), torch.stack(targets)
    return dataset

def postprocessing(origin, polluted, repair):
    origin = origin.view(1,1,1,-1)
    origin = torch.squeeze(origin)
    polluted = polluted.view(1,1,1,-1)
    polluted = torch.squeeze(polluted)
    repair = torch.squeeze(repair)

    for i in range(784):
        if origin[i]==polluted[i]:
            repair[i]=origin[i]
    
    repair = repair.view(1, -1)
    return repair