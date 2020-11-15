import torch

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return torch.stack(tuple(reversed(out))).t()