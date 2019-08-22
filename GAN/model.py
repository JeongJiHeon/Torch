import torch
import dcgan
import cgan
import pix2pix


def model(model):
    '''
           ========================================
           parameters
               DCGAN:
                   model = 'dcgan'
                   
                   return : Generator, Discriminator
           ========================================
    '''
    if model =='dcgan':
        Generator = dcgan.Generator
        Discriminator = dcgan.Discriminator

    elif model == 'cgan':
        Generator = cgan.Generator
        Discriminator = cgan.Discriminator
    
    elif model == 'pix2pix':
        Generator = pix2pix.Generator
        Discriminator = pix2pix.Discriminator
    
    return Generator, Discriminator



def optim(*args, **kwargs):
    '''
           ========================================
           parametrs
               args :
                   model which want training
               Adam Optimizer :
                   opt = 'adam', lr = 0.002, betas = (0.5, 0.999)
                   
           return
               Optimizer List
           ========================================
    '''
    
    if kwargs['opt'] == 'adam':
        return [torch.optim.Adam(m.parameters(), lr = kwargs['lr'], betas=kwargs['betas']) for m in args]
       
def loss(loss):
    '''
           ========================================
           parametrs
               loss = BCELoss
               
           return
               Loss Function
           ========================================
    '''
    if loss == 'BCELoss':
        return torch.nn.BCELoss()