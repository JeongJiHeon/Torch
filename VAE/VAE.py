import utils
import torch
import torchvision
import os
import numpy as np
from network import *
from DataLoader import *

class VAE(object):
    def __init__(self):
        self.step = 0
        
    def load_dataset(self, root, batch_size = 128, data = 'mnist'):
        self.dataloader = Public_Dataloader(root = root, batch_size = batch_size, data = data, t = transforms.Compose([transforms.Resize(28), transforms.ToTensor()]))
        self.batch_size = batch_size
        
    def build_model(self, lr, EPOCH):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.VAE = Varational_AutoEncoder().cuda(device = self.device)
            
        else:
            self.device = torch.device('cpu')
            self.VAE = Varational_AutoEncoder()
            
        self.optim = utils.optim(self.VAE, lr = lr)
        self.EPOCH = EPOCH
        
        if 'VAE.pkl' in os.listdir():
            self.VAE = torch.load('VAE.pkl')
            
    def train(self):
        try:
            getattr(self, 'VAE')
            getattr(self, 'optim')
        except:
            assert False, 'Not apply build_model'
            
        try:
            getattr(self, 'dataloader')
        except:
            assert False, 'Not apply load_dataset'
        for epoch in range(self.EPOCH):    
            for i, data in enumerate(self.dataloader):
                self.optim.zero_grad()
                x_hat, output, loss, mll, KL= self.VAE(data[0].to(self.device).reshape(self.batch_size,-1))
                loss.backward()
                self.optim.step()

                utils.PresentationExperience(epoch+1, i, 100, ELBO = -loss.item(), marginal_loss = mll ,KL = KL)
            if i % 100 == 99:
                torch.save("VAE.pkl", self.VAE)


            



