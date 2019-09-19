import torch
import VAE

def model(model):
    '''
           ========================================
           parameters
               Varational AutoEncoder:
                   model = 'VAE'
                   
                   return : VAE

           ========================================
    '''
    if model =='VAE':
        return VAE.Varational_AutoEncoder
