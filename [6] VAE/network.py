import torch
import torch.nn as nn
import torch.nn.functional as F

z_dim = 2

class Bernoulli_MLP_decoder(nn.Module):
    def __init__(self):
        super(Bernoulli_MLP_decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.Tanh(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,784),
        )
    def forward(self, inputs):
        return F.sigmoid(self.decoder(inputs))
    
    
class Bernoulli_MLP_encoder(nn.Module):
    def __init__(self):
        super(Bernoulli_MLP_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 100),
            nn.Tanh(),
            
            
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,z_dim*2),
        )
    def forward(self, inputs):
        output = self.encoder(inputs)
        return output[:,:z_dim], 1e-6 + F.softplus(output[:,z_dim:])
    
class Varational_AutoEncoder(nn.Module):
    def __init__(self):
        super(Varational_AutoEncoder, self).__init__()
        self.E = Bernoulli_MLP_encoder()
        self.D = Bernoulli_MLP_decoder()
    def forward(self, inputs):
        inputs = (inputs+1)/2
        x_hat = inputs
        noise = torch.randn(inputs.shape)/3
        x_hat += noise
        mean, std = self.E(x_hat)
        epsilon = torch.randn(inputs.shape[0],z_dim)
        z = mean + std*epsilon
        y = self.D(z)
        
        
        marginal_likelihood = torch.sum(inputs * torch.log(1e-7 + y) + (1-inputs) * torch.log(1-y + 1e-7),1)
        marginal_likelihood = torch.mean(marginal_likelihood)
        KL_divergence = 0.5*torch.sum(mean**2 + std**2 - torch.log(1e-7 + std**2) -1, 1)
        KL_divergence = torch.mean(KL_divergence)
        ELBO = marginal_likelihood - KL_divergence
        
        return x_hat, y, -ELBO, -marginal_likelihood, KL_divergence