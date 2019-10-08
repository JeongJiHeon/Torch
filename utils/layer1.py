import torch
import torch.nn as nn



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, conv = 'conv', kernel_size = 4, stride = 2, bias = True):
        super(ConvLayer, self).__init__()
        model = []
        if conv == 'conv':
            model.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                                   kernel_size = kernel_size, stride = stride, bias = bias))
        elif conv == 'convT':
            model.append(nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels,
                                            kernel_size = kernel_size, stride = stride, bias = bias))
            
        self.conv = nn.Sequential(*model)
        
    def forward(self, inputs):
        return self.conv(inputs)
    
class PaddingLayer(nn.Module):
    def __init__(self, padding_size, Padding = 'zero'):
        super(PaddingLayer, self).__init__()
        model = []
        if Padding == 'zero':
            model.append(nn.ZeroPad2d(padding_size))
        elif padding == 'reflect':
            model.append(nn.ReflectionPad2d(padding_size))
            
        self.padding = nn.Sequential(*model)
            
    def forward(self, inputs):
        return self.padding(inputs)
    
    
class NormLayer(nn.Module):
    def __init__(self, channels, Norm = 'batch'):
        super(NormLayer, self).__init__()
        model = []
        if Norm == 'batch':
            model.append(nn.BatchNorm2d(channels))
        elif Norm == 'instance':
            model.append(nn.InstanceNorm2d(channels))
            
        self.norm = nn.Sequential(*model)
            
    def forward(self, inputs):
        return self.norm(inputs)
    
class ActivationLayer(nn.Module):
    def __init__(self, activation_fn = 'ReLU', leaky_scope = 0.2):
        super(ActivationLayer, self).__init__()
        model = []
        
        if activation_fn == 'Sigmoid':
            model.append(nn.Sigmoid())
        elif activation_fn == 'ReLU':
            model.append(nn.ReLU(True))
        elif activation_fn == 'Tanh':
            model.append(nn.Tanh())
        elif activation_fn == 'LeakyReLU':
            model.append(nn.LeakyReLU(leaky_scope, inplace = True))
        elif activation_fn == 'None':
            pass    
        
        self.activation_fn = nn.Sequential(*model)
        
    def forward(self, inputs):
        return self.activation_fn(inputs)
    
    
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv = 'conv', kernel_size = 4, stride = 2, Norm = 'batch', Padding = 'zero', activation_fn = 'ReLU'):
        super(ConvBlock, self).__init__()
        model = [
            PaddingLayer(padding_size = kernel_size//2, Padding = Padding),
            ConvLayer(in_channels = in_channels, out_channels, conv='conv', kernel_size = kernel_size, stride = stride, bias = True),
            NormLayer(channels = out_channels, Norm = Norm),
            ActivationLayer(activation_fn = activation_fn)
        ]
        self.ConvBlock = nn.Sequential(*model)
        
    def forward(self, inputs):
        return self.ConvBlock(inputs)

        
class ResBlock(nn.Module):
    def __init__(self, channels, Norm = 'instance', kernel_size = 4, stride = 2):
        super(ResBlock, self).__init__()
        model = [
            PaddingLayer(padding_size = kernel_size//2, Padding = 'reflect'),
            ConvLayer(in_channels = channels, out_channels = channels, conv = 'conv', kernel_size = kernel_size, stride = stride, bias = True),
            NormLayer(channels = channels, Norm = Norm),
            Activation_fn('ReLU')
        ]
        model *= 2
        model.pop()
        self.ResBlock = nn.Sequential(*model)
        
    def forward(self, inputs):
        return self.ResBlock(inputs) + inputs

        
        
