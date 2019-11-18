import torch
import torch.nn as nn
import torch.utils as utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision
import os
import numpy as np
import imageio
from random import *


def Loss(loss = 'BCELoss'):
    if loss == 'BCELoss':
        return nn.BCELoss()
    elif loss == 'MSELoss':
        return nn.MSELoss()
    elif loss == 'L1Loss':
        return nn.L1Loss()


def optim(m, lr, optim = 'Adam', betas = (0.5, 0.99)):
    if optim == 'Adam':
        return torch.optim.Adam(m.parameters(), lr = lr, betas = betas)
    elif optim == 'SGD':
        return troch.optim.SGD(m.parameters(), lr = lr, momentum = betas[0])




        
def PresentationExperience(epoch,num , stop,  **kargw):
    sentence = '\r[ epoch : %d ] [ num : %d ]'
    for i in range(len(kargw)):
        sentence += '[ %s : %.4f ] '
    print(sentence % (epoch, num, *[i for a in kargw for i in [a, kargw[a]]] ) , end = ' ')
    if num % stop == 0:
        print()
        
def Accuracy(X, Y):
    correct = 0
    for i in range(X.size(0)):
        if X[i][Y[i]] == X[i].max():
            correct += 1
    return correct/X.size(0)
        
def init_weight(net, init_type='normal', init_gain=0.02):
    '''
           ========================================
           parameters
               mean = 0. sd = 0.02
               
           Conv = Normal(0, 0.02)
           BatchNorm = Normal(1.0, 0.02)
           ========================================
    '''
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('InstanceNorm2d') != -1:  # InstanceNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        
    return init_func(net)
    
