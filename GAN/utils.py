import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils as utils
import torchvision.utils as vutils

import os
import numpy as np
import matplotlib.pyplot as plot


def make_weights_function(mean = 0 , sd = 0.02):
    '''
           ========================================
           parametrs
               mean = 0. sd = 0.02
               
           Conv = Normal(0, 0.02)
           BatchNorm = Normal(1.0, 0.02)
           ========================================
    '''
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, mean, sd)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, sd)
            nn.init.constant_(m.bias.data, 0)
        
    return weights_init


def save_model(model, **kwargs):
    '''
           ========================================
           parametrs
               name = string
           ========================================
    '''
    if 'name' in kwags:
        torch.save(model, kwargs['name'])
    else:
        torch.save(model, model.__class__.__name__ + '.pkl')
    print(model.__class__.__name__ +" save !")

    
def load_model(model, weights_init = make_weights_function(), **kwargs):
    '''
           ========================================
           parametrs
               name = string
           ========================================
    '''
    if 'name' in kwags:
        name = kwargs['name']
    else:
        name = model.__class__.__name__ + '.pkl'
        
    if name in os.listdir():
        print(model.__class__.__name__ +" restore !")
        return torch.load(name)
    else:
        
        return model.apply(weights_init)
    
def imshow(image):
    plt.figure(figsize=(8,8))
    plt.title("Training Image")
    plt.imshow(np.transpose(vutils.make_grid(image[0][:64], padding=2, normalize=True).cpu(),(1,2,0)))
    
    
def ShowPlot(List1, List2, xlabel=False, ylabel=False, title=False):
    plt.figure(figsize=(10,5))
    if title:
        plt.title(title)
    plt.plot(List1, label=List1.__class__.__name__)
    plt.plot(List2, label=List2.__class__.__name__)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
    
def data_loader(dataroot, batch_size = 128, image_size = 64, num_workers = 2 , drop_last = True, shuffle = True, **kwargs):
    '''
           ========================================
           parametrs
               dataroot = data path , batch_size = 128, image_size = 64, num_workers = 2, drop_last = True, shuffle = True,
               transform = torchvision.transform.Compose([ "ransforms module" ])
           ========================================
    '''

    if 'transform' in kwargs:
        dataset = dset.ImageFolder(root = dataroot, transform =  kwargs['transform'])
    else:
        dataset = dset.ImageFolder(root = dataroot, transform =  transforms.Compose( [transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor() ]))

        
    dataloader = utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, drop_last = drop_last, shuffle = shuffle)



    return dataloader