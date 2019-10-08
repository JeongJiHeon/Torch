import torch
import torch.nn as nn
from layer import *




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        channel = 512
        G = []
        G.append(ConvBlock(in_channels = 100, out_channels = channel, norm_layer = 'batchnorm', padding_mode = 'None',
                           activation_fn = 'ReLU', conv = 'convT',  kernel_size = 4, stride = 1, padding = 0, bias = False))
        for i in range(3):
            G.append(ConvBlock(in_channels = channel, out_channels = channel//2, norm_layer = 'batchnorm', padding_mode = 'None',
                               activation_fn = 'ReLU', conv = 'convT', kernel_size = 4, stride = 1, padding = 1, bias = False))
            channel //= 2
            
        G.append(ConvBlock(in_channels = channel, out_channels = 3, norm_layer = 'None', padding_mode = 'None',
                           activation_fn = 'Tanh', conv = 'convT', kernel_size = 4, stride = 2, padding = 1, bias = False))
        
        self.G = nn.Sequential(*G)
#         self.G = nn.Sequential(
#             nn.ConvTranspose2d(in_channels = 100, out_channels = 512, kernel_size = 4, stride = 1, padding = 0, bias = False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(in_channels = 64, out_channels = 3, kernel_size = 4, stride = 2, padding = 1, bias = False),
#             nn.Tanh()
#         )

        
    def forward(self, inputs):
        return self.G(inputs)
    


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        D = []
        channel = 64
        D.append(ConvBlock(in_channels = 3, out_channels = channel, norm_layer = 'None', padding_mode = 'None',
                           activation_fn = 'LeakyReLU', conv = 'conv', kernel_size = 4, stride = 2, padding = 1, bias = False))
        for i in range(3):
            D.append(ConvBlock(in_channels = channel, out_channels = channel * 2, norm_layer = 'batchnorm', padding_mode = 'None',
                               activation_fn = 'LeakyReLU', conv = 'conv', kernel_size = 4, stride = 2, padding = 1, bias = False))
            channel *= 2
        D.append(ConvBlock(in_channels = channel, out_channels = 1, norm_layer = 'None', padding_mode = 'None',
                           activation_fn = 'Sigmoid', conv = 'conv', kernel_size = 4, stride = 1, padding = 0, bias = False))
        self.D = nn.Sequential(*D)
        
#         self.D = nn.Sequential(
#             nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
#             nn.LeakyReLU(0.2, inplace = True),
            
#             nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace = True),
            
#             nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace = True),
            
#             nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace = True),
            
#             nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, stride = 1, padding = 0, bias = False),
#             nn.Sigmoid()
            
#         )
        
    def forward(self, inputs):
        return self.D(inputs)
    

        







