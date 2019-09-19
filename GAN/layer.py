import torch
import torch.nn as nn
import utils

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer, padding_mode, activation_fn, bias = False, leaky_scope = 0.2, padding = 3):
        super(Linearock, self).__init__()
        model = []
        if padding_mode[0] == 'reflect':
            model.append(nn.ReflectionPad2d(padding_mode[1]))
        elif padding_mode[0] == 'zero':
            model.append(nn.ZeroPad2d(padding_mode[1]))
        elif padding_mode[0] == 'None':
            pass
        
        model.append(nn.Linear(in_dim, out_dim))
        
        if norm_layer == 'batchnorm':
            model.append(nn.BatchNorm2d(out_channels))
        elif norm_layer == 'instance':
            model.append(nn.InstanceNorm2d(out_channels))
        elif norm_layer == 'None':
            pass
        
        
        if activation_fn == 'Sigmoid':
            model.append(nn.Sigmoid())
        elif activation_fn == 'ReLU':
            model.append(nn.ReLU(True))
        elif activation_fn == 'Tanh':
            model.append(nn.Tanh())
        elif activation_fn == 'LeakyReLU':
            model.append(nn.LeakyReLU(leaky_scope))
        elif activation_fn == 'None':
            pass
        
        
        self.model = nn.Sequential(*model)
        utils.init_weight(self.model)
    def forward(self, inputs):
        return self.model(inputs)
        


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, padding_mode, activation_fn, conv, kernel_size=3, stride = 1, padding = 1, bias = False, leaky_scope = 0.2):
        super(ConvBlock, self).__init__()
        model = []
        if padding_mode[0] == 'reflect':
            model.append(nn.ReflectionPad2d(padding_mode[1]))
        elif padding_mode[0] == 'zero':
            model.append(nn.ZeroPad2d(padding_mode[1]))
        elif padding_mode[0] == 'None':
            pass
        
        if conv == 'conv':
            model.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias))
        elif conv == 'convT':
            model.append(nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = padding, bias = bias))
        elif conv == 'None':
            pass
            
            
        if norm_layer == 'batchnorm':
            model.append(nn.BatchNorm2d(out_channels))
        elif norm_layer == 'instance':
            model.append(nn.InstanceNorm2d(out_channels))
        elif norm_layer == 'None':
            pass
            
            
        if activation_fn == 'Sigmoid':
            model.append(nn.Sigmoid())
        elif activation_fn == 'ReLU':
            model.append(nn.ReLU(True))
        elif activation_fn == 'Tanh':
            model.append(nn.Tanh())
        elif activation_fn == 'LeakyReLU':
            model.append(nn.LeakyReLU(leaky_scope))
        elif activation_fn == 'None':
            pass
            
        self.model = nn.Sequential(*model)
    def forward(self, inputs):
        return self.model(inputs)
    
    
    
class ResidualBlock(nn.Module):
    def __init__(self, dim, padding_mode,  norm_layer, kernel_size = 3, stride = 1):
        super(ResidualBlock, self).__init__()
        model = []
        if padding_mode == 'reflect':
            model.append(nn.ReflectionPad2d(1))
        elif padding_mode == 'zero':
            model.append(nn.ZeroPad2d(1))

        if norm_layer == 'batchnorm':
            norm = nn.BatchNorm2d(dim)
        elif norm_layer == 'instance':
            norm = nn.InstanceNorm2d(dim)

        model += [nn.Conv2d(dim, dim , kernel_size = kernel_size, stride = stride, bias = False), norm, nn.ReLU(True)]
        model *= 2
        model.pop()
        
        self.model = nn.Sequential(*model)
    def forward(self, inputs):
        return inputs + self.model(inputs)