import layer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import *



class Generator(nn.Module):
    def __init__(self, residual_block_num = 6):
        super(Generator, self).__init__()
        dim = 32
        G = []
        G.append(layer.ConvBlock(in_channels = 3, out_channels = dim, padding_mode = ('reflect',3), norm_layer = 'instance', 
                                activation_fn = 'ReLU', conv = 'conv', kernel_size = 7, stride = 1, padding = 0, bias = False))
        for i in range(2):
            G.append(layer.ConvBlock(in_channels = dim, out_channels = dim*2, padding_mode = ('zero',0), norm_layer = 'instance',
                                    activation_fn = 'ReLU', conv = 'conv', kernel_size = 3, stride = 2, padding = 1, bias = False))
            dim *= 2
        
        for i in range(residual_block_num):
            G.append(layer.ResidualBlock(dim = dim, padding_mode = ('reflect',1), norm_layer = 'instance', activation_fn = 'ReLU'))
            
        for i in range(2):
            G.append(layer.ConvBlock(in_channels = dim, out_channels = dim//2, padding_mode = ('zero',0), norm_layer = 'instance',
                                    activation_fn = 'ReLU', conv = 'convT', kernel_size = 3, stride = 2, padding = 1, bias = False))
            dim = dim//2
        G.append(layer.ConvBlock(in_channels = dim, out_channels = 3, padding_mode = ('reflect',3), norm_layer = 'instance',
                                activation_fn = 'Tanh', conv = 'conv', kernel_size = 7, stride = 1, padding = 0, bias = False))
        
        self.G = nn.Sequential(*G)
    def forward(self, inputs):
        return self.G(inputs)
    

class Discriminator(nn.Module):
    def __init__(self, norm_layer_num = 2):
        super(Discriminator, self).__init__()
        D = []
        dim = 64
        D.append(layer.ConvBlock(in_channels = 3, out_channels = dim, padding_mode = ('zero',0), norm_layer = 'None',
                                activation_fn = 'LeakyReLU', conv = 'conv', kernel_size = 4, stride = 2, padding = 1, bias = False))
        for i in range(norm_layer_num):
            D.append(layer.ConvBlock(in_channels = dim, out_channels = dim*2, padding_mode = ('zero',0), norm_layer = 'batchnorm',
                                    activation_fn = 'LeakyReLU', conv = 'conv', kernel_size = 4, stride = 2, padding = 1, bias = False))
            dim *= 2
        D.append(layer.ConvBlock(in_channels = dim, out_channels = dim*2, padding_mode = ('zero',0), norm_layer = 'batchnorm',
                    activation_fn = 'LeakyReLU', conv = 'conv', kernel_size = 4, stride = 1, padding = 1, bias = False))
        D.append(layer.ConvBlock(in_channels = dim*2, out_channels = 1, padding_mode = ('zero',0), norm_layer = 'None',
                                activation_fn = 'None', conv = 'conv', kernel_size = 4, stride = 1, padding = 1, bias = False))      
        
        self.D = nn.Sequential(*D)

    def forward(self, inputs):
        outputs = self.D(inputs)
        return F.avg_pool2d(outputs, outputs.size()[2:]).view(outputs.size()[0], -1)
    
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        

