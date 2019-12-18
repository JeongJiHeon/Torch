import torch
import torchvision
import torch.nn as nn
import layer
import torchvision.transforms as transforms
import utils
import os

from network import *
from DataLoader import *

class DCGAN(object):
    def __init__(self):
        
    def load_dataset(self, root, batch_size = 64, data = 'cifar10'):
        self.dataloader = Public_Dataloader(root = root, batch_size = batch_size, data = data)
    
    def build_model(self, lr, Epoch, loss = 'BCELoss', optim = 'Adam', betas = (0.5, 0.999)):
        
        self.Epoch = Epoch
        #### Build Model ####
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.Generator = Generator().cuda(self.device)
            self.Discriminator = Discriminator().cuda(self.device)
        else:
            self.device = torch.device('cpu')
            self.Generator = Generator()
            self.Discriminator = Discriminator()
        
        if 'Generator.pkl' in os.listdir():
            self.Generator = torch.load('Generator.pkl')
        if 'Discriminator.pkl' in os.listdir():
            self.Discriminator = torch.load('Discriminator.pkl')


        #### Build Loss Function ####
#        self.criterion = utils.Loss(loss = loss)
        
        #### Build optimizing ####
        self.optimG = utils.optim(self.Generator, lr = lr, optim = optim, betas = betas)
        self.optimD = utils.optim(self.Discriminator, lr = lr, optim = optim, betas = betas)
                 
                 
    def train(self):
        #### Check build_model ####
        try:
            getattr(self, 'Generator')
            getattr(self, 'Discriminator')
#            getattr(self, 'criterion')
            getattr(self, 'optimG')
            getattr(self, 'optimD')
        except:
            assert False, 'Not apply build_model'
        
        #### Check load_dataset ####
        try:
            getattr(self, 'dataloader')
        except:
            assert False, 'Not apply load_dataset'
            
        one = torch.FloatTensor([1]).to(device = self.device)
        mone = one * -1
        #### Train ####    
        for epoch in range(self.Epoch):
#            if epoch < self.step:
#                continue
            for i, data in enumerate(self.dataloader):
                latent = torch.randn(data[0].size(0), 100, 1, 1).to(device = self.device)
                                
                #### train Discriminator ####
                self.optimD.zero_grad()
                real = data[0].to(self.device)
                fake = self.Generator(latent)
                real_D = self.Discriminator(real)
                fake_D = self.Discriminator(fake)
                      
                
                loss_real = real_D.mean() * one
                loss_fake = fake_D.mean() * mone
                
                loss_real.backward()
                loss_fake.backward()
                
                lossD = loss_real - loss_fake

                self.optimD.step()
                
                #### train Generator ####
                latent = torch.randn(data[0].size(0), 100, 1, 1).to(device = self.device)
                fake = self.Generator(latent)
                fake_D = self.Discriminator(fake)
                
                self.optimG.zero_grad()
                lossG = fake_D.mean() * mone
                lossG.backward()
                lossG *= -1
                self.optimG.step()
                
                
                utils.PresentationExperience(epoch, i, 100, lossG = lossG.item(), lossD = lossD.item())
                if i % 100 == 0:
                    torch.save(self.Generator, 'Generator.pkl')
                    torch.save(self.Discriminator, 'Discriminator.pkl')
            