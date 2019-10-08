import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils as utils
import torchvision.utils as vutils
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from random import *

def dataset(data = 'mnist', root = 'data', train = True, download = True, size = 28, transform = True, 
            batch_size = 128, shuffle = True, num_workers = 2, drop_last = True):
    '''
           ========================================
           parameters
               data(str) = 'mnist' (default) or 'cifar10' 
               root(str) = Download directory or dataset directory
               train(bool) = True(default)
               download(bool) = True(default)
               size(int) = 28(default)
               transform(torchvision.transforms.Compose) = Resize(size, size), ToTensor()
               batch_size(int) = 128(default)
               shuffle(bool) = True(default)
               num_workers(int) = 2(default)
           ========================================
    '''
    if transform:
        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
                                                ])
    else:
        _transform = transform
    
    if not root in os.listdir():
        os.mkdir(root)
    root = os.getcwd()+ '/' + root


        
    
    if data == 'mnist':
        trainset = torchvision.datasets.MNIST(root= root, train=True, transform=_transform, download=True)
    elif data == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root= root, train=True, transform=_transform, download=True)


    
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, drop_last = drop_last)
    return trainloader


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


def save_model(model, **kwargs):
    '''
           ========================================
           parameters
               name = string
           ========================================
    '''
    if 'name' in kwargs:
        torch.save(model, kwargs['name'])
    else:
        torch.save(model, model.__class__.__name__ + '.pkl')
    print(model.__class__.__name__ +" save !")

    
def load_model(model, **kwargs):
    '''
           ========================================
           parameters
               name = string
           ========================================
    '''
    if 'name' in kwargs:
        name = kwargs['name']
    else:
        name = model.__class__.__name__ + '.pkl'        
    if name in os.listdir():
        print(model.__class__.__name__ +" restore !")
        return torch.load(name)
    else:
        return model
    
def imshow(image):
    plt.figure(figsize=(8,8))
    plt.title("Training Image")
    plt.imshow(np.transpose(vutils.make_grid(torch.Tensor(image)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    
    
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
           parameters
               dataroot = data path , batch_size = 128, image_size = 64, num_workers = 2, drop_last = True, shuffle = True,
               transform = torchvision.transform.Compose([ "ransforms module" ])
           ========================================
    '''

    if 'transform' in kwargs:
        dataset = dset.ImageFolder(root = dataroot, transform =  kwargs['transform'])
    else:
        dataset = dset.ImageFolder(root = dataroot, transform =  transforms.Compose( 
            [transforms.Resize(image_size), 
             transforms.CenterCrop(image_size), 
             transforms.ToTensor(), 
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ))

        
    dataloader = utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, drop_last = drop_last, shuffle = shuffle)



    return dataloader

def saveimage(imglist, path, figsize = (6,6), x=8, y=8, step_num = 10, label = True, name = ''):
    for num in range(len(imglist)):
        if num % step_num != 0:
            continue
        path =path.format(num)

        fig, ax = plt.subplots(x, y, figsize=figsize)
        num_img = imglist[num]
        for i in range(len(num_img)):
            ax[int(i/8), i%8].imshow(num_img[i].reshape(28,28), cmap='gray')
            ax[int(i/8), i%8].get_xaxis().set_visible(False)
            ax[int(i/8), i%8].get_yaxis().set_visible(False)
            
        if name == '':
            name = 'Epoch {0}'.format(num)
        if label:
            fig.text(0.5, 0.04, name, ha='center', fontsize = 15)
        plt.savefig(path)
        plt.show()
        plt.close()

def makegif(imglist, path, name,  num = 1):
    images = []
    for e in range(len(imglist)):
        img_name = path + str(e * num) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(name, images, fps = 5)

    
def ImagePool(pool, image, max_size = 50):
    gen_image = image.detach().numpy()
    if len(pool) < max_size:
        pool.append(gen_image)
        return image
    else:
        p = random()
        if p > 0.5:
            random_id = randint(0, len(pool)-1)
            temp = pool[random_id]
            pool[random_id] = gen_image
            return torch.Tensor(temp), True
        else:
            return torch.Tensor(gen_image), False
        
    
    
    
    
    
    
    