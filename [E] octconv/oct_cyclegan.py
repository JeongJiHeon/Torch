import layer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import *
import torchvision.transforms as transforms


class Generator(nn.Module):
    def __init__(self, residual_block_num = 9):
        super(Generator, self).__init__()
        dim = 32
        G = []
        G.append(OctConv2d())
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
                                    activation_fn = 'ReLU', conv = 'convT', kernel_size = 3, stride = 2, padding = 1, bias = True))
            dim = dim//2
        G.append(layer.ConvBlock(in_channels = dim, out_channels = 3, padding_mode = ('reflect',3), norm_layer = 'instance',
                                activation_fn = 'Tanh', conv = 'conv', kernel_size = 7, stride = 1, padding = 0, bias = True))
        
        self.G = nn.Sequential(*G)
    def forward(self, inputs):
        return self.G(inputs)
    

class Discriminator(nn.Module):
    def __init__(self, norm_layer_num = 3, patch = True, a = 0.5):
        super(Discriminator, self).__init__()            
        D = []
        dim = 64
        self.patch = patch
        self.a = a

        if self.patch:
            D.append(OctConv2d(in_channels = 3, out_channels = dim, kernel_size = 4, a_in = 0, a_out = a, stride = 2, padding = 1, bias = True,
                               act = 'leakyrelu', norm = 'none'))
            D.append(OctConv2d(in_channels = dim, out_channels = dim*2, kernel_size = 4, a_in = a, a_out = a, stride = 2, padding = 1, bias = True,
                               act = 'leakyrelu', norm = 'batch'))
            D.append(OctConv2d(in_channels = dim*2, out_channels = dim*4, kernel_size = 4, a_in = a, a_out = 0, stride = 2, padding = 1, bias = True,
                               act = 'leakyrelu', norm = 'batch'))
            
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

        

class OctConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, a_in = 0.25, a_out = 0.25, stride = 1, padding = 0, bias = True, dilation = 1, groups = 1, padding_mode = 'zero', act = 'relu', norm = 'batch'):
        super(OctConv2d, self).__init__()

        assert 0 <= a_in <= 1, ' a_in must be in [0,1]'
        assert 0 <= a_out <=1, 'a_out must be in [0,1]'
        
        assert in_channels * a_in - int(in_channels * a_in) == 0, 'not int'
        assert out_channels * a_out - int(out_channels * a_out) == 0, 'not int'
        
        self.in_channels_H = int(in_channels * (1-a_in))
        self.out_channels_H = int(out_channels * (1-a_out))
        
        self.a_in = a_in
        self.a_out = a_out
        self.norm = True
        self.act = True
        HH, HL, LH, LL = None, None, None, None
        
        if a_in == 0:
            HH = nn.Conv2d(self.in_channels_H, self.out_channels_H, kernel_size = kernel_size, 
                           stride= stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
            HL = nn.Conv2d(self.in_channels_H, out_channels - self.out_channels_H, kernel_size = kernel_size, 
                           stride= stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
        
        elif a_out == 0:
            HH = nn.Conv2d(self.in_channels_H, self.out_channels_H, kernel_size = kernel_size, 
                            stride= stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
            LH = nn.Conv2d(in_channels - self.in_channels_H, self.out_channels_H, kernel_size = kernel_size,
                            stride = stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
            
        elif a_in == 0 and a_out == 0:
            HH = nn.Conv2d(self.in_channels_H, out_channels_H, kernel_size = kernel_size, 
                            stride= stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)

        else:
            HH = nn.Conv2d(self.in_channels_H, self.out_channels_H, kernel_size = kernel_size, 
                          stride= stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
            HL = nn.Conv2d(self.in_channels_H, out_channels - self.out_channels_H, kernel_size = kernel_size, 
                          stride= stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
            LL = nn.Conv2d(in_channels - self.in_channels_H, out_channels - self.out_channels_H, kernel_size = kernel_size,
                          stride = stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
            LH = nn.Conv2d(in_channels - self.in_channels_H, self.out_channels_H, kernel_size = kernel_size,
                          stride = stride, padding = padding, bias = bias, dilation = dilation, groups = groups, padding_mode = padding_mode).to(device)
            
            
        self.W = [HH, HL, LH, LL]
        
        if act == 'relu':
            self.Activation = nn.ReLU()
        elif act == 'sigmoid':
            self.Activation = nn.Sigmoid()
        elif act == 'leakyrelu':
            self.Activation = nn.LeakyReLU()
        else:
            self.act = False
            
        if norm == 'batchnorm':
            self.Normalization = nn.BatchNorm2d
        elif norm == 'instance':
            self.Normalization = nn.InstanceNorm2d
        else:
            self.norm = False
        
    def forward(self, inputs):
        X_H , X_L = (inputs[:, :self.in_channels_H], inputs[:, self.in_channels_H:]) if type(inputs) != tuple else inputs
        X = [X_H.to(device), F.max_pool2d(X_H,2).to(device), X_L.to(device), X_L.to(device)]

        Y = []
        for num, weight in enumerate(self.W):
            Y.append(0 if type(weight) == type(None) else weight(X[num]))

        if type(Y[2]) != int:
            Y[2] = nn.Upsample(scale_factor=2, mode = 'nearest')(Y[2])
        
        if not self.act:
            return Y[0]+Y[2], Y[1]+Y[3]
        Y_H = self.Activation(Y[0] + Y[2]) if isinstance(Y[0] + Y[2], torch.Tensor) else 0
        Y_L = self.Activation(Y[1] + Y[3]) if isinstance(Y[1] + Y[3], torch.Tensor) else 0
        
        if not self.norm:
            return Y_H, Y_L                           
        Y_H = self.Normalization(Y_H.size(1)).to(device)(Y_H) if isinstance(Y_H, torch.Tensor) else 0
        Y_L = self.Normalization(Y_L.size(1)).to(device)(Y_L) if isinstance(Y_L, torch.Tensor) else 0
        
        
        
        return (Y_H, Y_L) if self.a_out != 0 else Y_H