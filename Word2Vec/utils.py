import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EMB_DIM = 32
RANGE = 2