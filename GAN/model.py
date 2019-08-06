import torch
import dcgan




def model(model):
    if model =='dcgan':
        Generator = dcgan.Generator
        Discriminator = dcgan.Discriminator

    return Generator, Discriminator



def optim(*args, **kwargs):
    
    if kwargs['opt'] == 'adam':
        return [torch.optim.Adam(m.parameters(), lr = kwargs['lr'], betas=kwargs['betas']) for m in args]
       
def loss(loss):
    if loss == 'BCELoss':
        return torch.nn.BCELoss()