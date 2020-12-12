from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn

criterion = nn.BCEWithLogitsLoss()
n_epochs = 200 
batch_size = 128
lr = 0.00001

def get_dataloader(batch_size):
    dataloader = DataLoader(MNIST('.',transform=transforms.ToTensor(),batch_size=batch_size,shuffle=True))
    return dataloader

