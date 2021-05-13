import torch

def preprocess(data):
    # Format the data to be in [0, 1]
    return data.float() / 255.0

def unprocess(data):
    # Transform the data from [0, 1] to [0, 255]
    return torch.clamp(data * 255.0, 0.0, 255.0).to(torch.uint8)