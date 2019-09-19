import torch
import torch.nn as nn
from layer import *



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        z, y, G = [], [], []
        
        # z_input Layer
        z.append(LinearBlock(in_dim = 100, out_dim = 256, norm_layer = 'batchnorm', padding_mode = 'None', activation_fn = 'ReLU'))
        
        # y_input Layer
        y.append(LinearBlock(in_dim = 10, out_dim = 256, norm_layer = 'batchnorm', padding_mode = 'None', activation_fn = 'ReLU'))

        # Generator Layer
        G.append(LinearBlock(in_dim = 512, out_dim = 1024, norm_layer = 'batchnorm', padding_mode = 'None', activation_fn = 'ReLU'))
        G.append(LinearBlock(in_dim = 1024, out_dim = 784*2, norm_layer = 'batchnorm', padding_mode = 'None', activation_fn = 'ReLU'))
        G.append(LinearBlock(in_dim = 784*2, out_dim = 784, norm_layer = 'None', padding_mode = 'None', activation_fn = 'Tanh'))
        self.z_input = nn.Sequential(*z)
        self.y_input = nn.Sequential(*y)
        self.G = nn.Sequential(*G)
        
        
        
#         # Latent 100
#         self.z_input = nn.Sequential(
#             nn.Linear(100, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(True)
#         )
        
#         # Condition 10
#         self.y_input = nn.Sequential(
#             nn.Linear(10,256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(True)
#         )
#         # Concat
#         self.G = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
            
#             nn.Linear(1024, 784*2),
#             nn.BatchNorm1d(784*2),
#             nn.ReLU(True),
            
#             nn.Linear(784*2, 784),
#             nn.Tanh()
#         )
        
    def forward(self, z, y):
        Z = self.z_input(z)
        Y = self.y_input(y)
        output = self.G(torch.cat([Z, Y], 1))
        return output

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        x, y, D = [], [], []
        # x_input Layer
        x.append(LinearBlock(in_dim = 784, out_dim = 500, norm_layer = 'None', padding_mode = 'None', activation_fn = 'LeakyReLU'))
        
        # y_input Layer
        y.append(LinearBlock(in_dim = 10, out_dim = 500, norm_layer = 'None', padding_mode = 'None', activation_fn = 'LeakyReLU'))
        
        # Discriminaotr Layer
        D.append(LinearBlock(in_dim = 1000, out_dim = 512, norm_layer = 'batchnorm', padding_mode = 'None', activation_fn = 'LeakyReLU'))
        D.append(LinearBlock(in_dim = 512, out_dim = 256, norm_layer = 'batchnorm', padding_mode = 'None', activation_fn = 'LeakyReLU'))
        D.append(LinearBlock(in_dim = 256, out_dim = 1, norm_layer = 'batchnorm', padding_mode = 'None', activation_fn = 'Sigmoid'))
        
        
        self.x_input = nn.Sequential(*x)
        self.y_input = nn.Sequential(*y)
        self.D = nn.Sequential(*D)



        
#         self.x_input = nn.Sequential(
#             nn.Linear(784, 500),
#             nn.LeakyReLU(0.2),
#         )
        
#         self.y_input = nn.Sequential(
#             nn.Linear(10, 500),
#             nn.LeakyReLU(0.2),
#         )
        
#         self.D = nn.Sequential(
#             nn.Linear(1000, 512),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(0.2),
            
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.2),
            
            
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
        
    def forward(self, x, y):
        X = self.x_input(x)
        Y = self.y_input(y)
        output = self.D(torch.cat([X, Y], 1))
        return output

             
    
    
    
    
    
    
    
    
    