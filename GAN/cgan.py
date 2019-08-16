import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    -`Generator `
        - Z Layer - ( noise )
            - input size =  (BatchSize, 100)
            - output size = (BatchSize, 256)
        - Y Layer - ( condi )
            - input size = (Batchsize, 10)
            - output size = (Batchsize, 256)
        - Layer1 - ( concat )
            - input size = (Batchsize, 512)
            - output size = (Batchsize, 1024)
        - Layer2 - ( concat )
            - input size = (Batchsize, 1024)
            - output size = (Batchsize, 784*2)
        - Layer1 - ( concat )
            - input size = (Batchsize, 784*2)
            - output size = (Batchsize, 784)
    """
    
    def __init__(self):
        super(Generator, self).__init__()
        # Latent 100
        self.z_input = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )
        # Condition 10
        self.y_input = nn.Sequential(
            nn.Linear(10,256),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )
        # Concat
        self.G = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 784*2),
            nn.BatchNorm1d(784*2),
            nn.ReLU(True),
            
            nn.Linear(784*2, 784),
            nn.Tanh()
        )
        
    def forward(self, z, y):
        Z = self.z_input(z)
        Y = self.y_input(y)
        output = self.G(torch.cat([Z, Y], 1))
        return output

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.x_input = nn.Sequential(
            nn.Linear(784, 500),
            nn.LeakyReLU(0.2),
        )
        
        self.y_input = nn.Sequential(
            nn.Linear(10, 500),
            nn.LeakyReLU(0.2),
        )
        
        self.D = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, y):
        X = self.x_input(x)
        Y = self.y_input(y)
        output = self.D(torch.cat([X, Y], 1))
        return output

             
    
    
    
    
    
    
    
    
    