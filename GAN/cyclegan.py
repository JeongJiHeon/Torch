import layer
import torch
import torch.nn as nn



class Generator(nn.Module):
    def __init__(self, residual_block_num = 9):
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
            G.append(layer.ResidualBlock(dim = dim, padding_mode = 'reflect', norm_layer = 'instance'))
            
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
    def __init__(self):
        super(Discriminator, self).__init__()
        D = []
        dim = 64
        D.append(layer.ConvBlock(in_channels = 3, out_channels = dim, padding_mode = ('zero',0), norm_layer = 'None',
                                activation_fn = 'LeakyReLU', conv = 'conv', kernel_size = 4, stride = 2, padding = 1, bias = False))
        for i in range(2):
            D.append(layer.ConvBlock(in_channels = dim, out_channels = dim*2, padding_mode = ('zero',0), norm_layer = 'instance',
                                    activation_fn = 'LeakyReLU', conv = 'conv', kernel_size = 4, stride = 2, padding = 1, bias = False))
            dim *= 2
        D.append(layer.ConvBlock(in_channels = dim, out_channels = dim*2, padding_mode = ('zero',0), norm_layer = 'None',
                                activation_fn = 'Sigmoid', conv = 'conv', kernel_size = 4, stride = 2, padding = 1, bias = False))      
        
        self.D = nn.Sequential(*D)
    def forward(self, inputs):
        return self.D(inputs)