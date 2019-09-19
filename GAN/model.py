import torch
import dcgan
import cgan
import pix2pix
import cyclegan

def model(model):
    '''
           ========================================
           parameters
               DCGAN:
                   model = 'dcgan'
                   
                   return : Generator, Discriminator
                   
               CGAN:
                   model = 'cgan'
                   
                   return : Generator, Discriminator
                   
               pix2pix:
                   model = 'pix2pix'
                   
                   return : Generator, Discriminaotr
                   
               CycleGAN:
                   model = 'cyclegan'
                   
                   return Generator, Discriminator
           ========================================
    '''
    if model =='dcgan':
        return dcgan.Generator, dcgan.Discriminator

    elif model == 'cgan':
        return cgan.Generator, cgan.Discriminator
    
    elif model == 'pix2pix':
        return pix2pix.Generator, pix2pix.Discriminator
    
    elif model == 'cyclegan':
        return cyclegan.Generator, cyclegan.Discriminator



