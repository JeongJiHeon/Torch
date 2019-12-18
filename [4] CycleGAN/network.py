import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import *
import torchvision.transforms as transforms
import layer


class Generator(nn.Module):
    def __init__(self, residual_block_num = 9):
        super(Generator, self).__init__()
        dim = 32
        G = []
        G.append(layer.ConvBlock(in_channels = 3, out_channels = dim, padding_mode = ('reflect',3), norm_layer = 'instance', 
                                activation_fn = 'ReLU', conv = 'conv', kernel_size = 7, stride = 1, padding = 0, bias = True))
        for i in range(2):
            G.append(layer.ConvBlock(in_channels = dim, out_channels = dim*2, padding_mode = ('zero',0), norm_layer = 'instance',
                                    activation_fn = 'ReLU', conv = 'conv', kernel_size = 3, stride = 2, padding = 1, bias = True))
            dim *= 2
        
        for i in range(residual_block_num):
            G.append(layer.ResidualBlock(dim = dim, padding_mode = ('reflect',1), norm_layer = 'instance', activation_fn = 'ReLU'))
            
        for i in range(2):
            G.append(layer.ConvBlock(in_channels = dim, out_channels = dim//2, padding_mode = ('zero',0), norm_layer = 'instance',
                                    activation_fn = 'ReLU', conv = 'convT', kernel_size = 4, stride = 2, padding = 1, bias = True))
            dim = dim//2
        G.append(layer.ConvBlock(in_channels = dim, out_channels = 3, padding_mode = ('reflect',3), norm_layer = 'instance',
                                activation_fn = 'Tanh', conv = 'conv', kernel_size = 7, stride = 1, padding = 0, bias = True))
        
        self.G = nn.Sequential(*G)
    def forward(self, inputs):
        return self.G(inputs)
    

class Discriminator(nn.Module):
    def __init__(self, norm_layer_num = 3, patch = True):
        super(Discriminator, self).__init__()            
        D = []
        dim = 64
        self.Patch = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(70), transforms.ToTensor()])
        self.patch = patch

        if self.patch:
            D.append(layer.ConvBlock(in_channels = 3, out_channels = dim, padding_mode = ('zero', 0), norm_layer = 'None',
                                     activation_fn = 'LeakyReLU', conv = 'conv', kernel_size = 4, stride = 2, padding = 1, bias = True))
            
            D.append(layer.ConvBlock(in_channels = dim, out_channels = dim*2, padding_mode = ('zero',0), norm_layer = 'batchnorm',
                                     activation_fn = 'LeakyReLU', conv = 'conv', kernel_size = 4, stride = 2, padding = 1, bias = True))
            
            D.append(layer.ConvBlock(in_channels = dim*2, out_channels = dim*4, padding_mode = ('zero',0), norm_layer = 'batchnorm',
                                     activation_fn = 'LeakyReLU', conv = 'conv', kernel_size = 4, stride = 2, padding = 1, bias = True))
            
            D.append(layer.ConvBlock(in_channels = dim*4, out_channels = dim*8, padding_mode = ('zero',0), norm_layer = 'batchnorm',
                                     activation_fn = 'LeakyReLU', conv = 'conv', kernel_size = 4, stride = 1, padding = 1, bias = True))
            
            D.append(layer.ConvBlock(in_channels = dim*8, out_channels = 1, padding_mode = ('zero', 0), norm_layer = 'None',
                                     activation_fn = 'None', conv = 'conv', kernel_size = 4, stride = 1, padding = 1, bias = True))

        self.D = nn.Sequential(*D)

    def forward(self, inputs):
#        inputs = self.Patch(inputs[0]).resize(1, 3, 70,70)
        outputs = self.D(inputs)
        if self.patch:
            return outputs

    
    
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        

