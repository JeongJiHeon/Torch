import torch
import torchvision
import torch.utils as utils
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os


EPOCHS = 5
batch_size = 128
fix_latent = torch.randn(64, 100, 1, 1)
dataroot = '/Users/mac/python/dataset'







class Generator(nn.Module):
    """
    -`Generator (UPSAMPLING)`
        - `ConvT Layer 1 - `
            - input size = (BatchSize , 100, 1)
            - output size = (BatchSize, 512, 4, 4)
        - `ConvT Layer 2 - `
            - input size = (BatchSize , 512, 4, 4)
            - output size = (BatchSize, 256, 8, 8)
        - `ConvT Layer 3 - `
            - input size = (BatchSize , 256, 8, 8)
            - output size = (BatchSize, 128, 16, 16)
        - `ConvT Layer 4 - `
            - input size = (BatchSize , 128, 16, 16)
            - output size = (BatchSize, 64, 32, 32)
        - `ConvT Layer 5 - `
            - input size = (BatchSize, 64, 32, 32)
            - output size = (BatchSize, 3, 64, 64)
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.G = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 100, out_channels = 512, kernel_size = 4, stride = 1, padding = 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels = 64, out_channels = 3, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )
        
    def forward(self, inputs):
        return self.G(inputs)
    


class Discriminator(nn.Module):
    """
    -`Discriminator (DOWNSAMPLING)`
        - `Conv Layer 1 - `
            - input size = (BatchSize , 3, 64, 64)
            - output size = (BatchSize, 64, 32, 32)
        - `Conv Layer 2 - `
            - input size = (BatchSize , 64, 32, 32)
            - output size = (BatchSize, 128, 16, 16)
        - `Conv Layer 3 - `
            - input size = (BatchSize , 128, 16, 16)
            - output size = (BatchSize, 256, 8, 8)
        -`Conv Layer 4 - `
            -input size = (BatchSize , 256, 8, 8)
            -output size = (BatchSize, 1)
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, stride = 1, padding = 0),
            nn.Sigmoid()
            
        )
        
    def forward(self, inputs):
        return self.D(inputs)
    

        







