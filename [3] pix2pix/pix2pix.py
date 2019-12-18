#-*- coding: utf-8 -*-
    
from network import *
from DataLoader import *

import torch
import numpy as np
import utils
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os


#datarootA = '/root/deeplearning/ukiyoe2photo/trainA'
#datarootB = '/root/deeplearning/ukiyoe2photo/trainB'

#datarootA = '/Users/mac/Documents/GitHub/GAN/google/data/ukiyoe2photo/trainA'
#datarootB = '/Users/mac/Documents/GitHub/GAN/data/ukiyoe2photo/trainB'

import time


class PIX2PIX(nn.Module):
    def __init__(self):
        super(PIX2PIX, self).__init__()
        self.step = 0
    def load_dataset(self, root, batch_size = 1):
        self.dataloader = Dataloader(root = root[0], batch_size = batch_size)
        
    def build_model(self, lr, EPOCH, loss = 'MSELoss', optim = 'Adam', betas = (0.5, 0.999), cycleLambda = 10):
        self.EPOCH = EPOCH
        self.cycleLambda = cycleLambda
        self.lr = lr
        self.optim = optim
        self.betas = betas
        
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
        
        self.criterion = utils.Loss(loss = loss)
        self.L1Loss = utils.Loss(loss = 'L1Loss')

    def train(self):
        #### Check build_model ####
        try:
            getattr(self, 'Generator')

            getattr(self, 'Discriminator')

            getattr(self, 'criterion')
            getattr(self, 'L1Loss')

        except:
            assert False, 'Not apply build_model'
        
        #### Check load_dataset ####
        try:
            getattr(self, 'dataloader')

        except:
            assert False, 'Not apply load_dataset'
        
        
        for epoch in range(self.EPOCH):
            if epoch < self.step:
                continue
            if epoch >= 100:
                lr = 0.0002 - 0.0002*(epoch-100)/100
                
            for i, data in enumerate(zip(self.dataloader)):
                real = data[0].to(self.device)
                
                self.optimG = utils.optim(self.Generator, lr = self.lr, optim = self.optim, betas =self. betas)
                self.optimD = utils.optim(self.Discriminator, lr = self.lr, optim = self.optim, betas = self.betas)
                
                fake = self.GeneratorA(real)
                

                D_real = self.DiscriminatorA(real)
                
                D_fake = self.DiscriminatorA(fake)
                
                
                ### train Generator ###
                self.optimG.zero_grad()
                
                LossG = self.criterion(D_fake, torch.ones(D_fake.shape).to(self.device))
                LossG.backward(retain_graph = True)
                
                self.optimG.step()

                
                ### train Discriminator ###
                
            
                self.optimD.zero_grad()
                
                
                
                loss_real = self.criterion(D_real, torch.ones(D_real.shape).to(self.device))
                loss_fake = self.criterion(D_fake,torch.zeros(D_fake.shape).to(self.device))
                
                (lossA_real + lossA_fake).backward(retain_graph = True)
                
                self.optimD.step()

                utils.PresentationExperience(epoch, i, 100, G = LossG.item(), D = (loss_real+loss_fake).item())
                
                if i % 100 == 99:
                    torch.save(self.Discriminator, 'Discriminator.pkl')
                    torch.save(self.Generator, 'Generator.pkl')
