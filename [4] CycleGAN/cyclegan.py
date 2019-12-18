#-*- coding: utf-8 -*-
    
from network import *
from DataLoader import *
from ImagePool import *

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


class CYCLEGAN(nn.Module):
    def __init__(self):
        super(CYCLEGAN, self).__init__()
        self.step = 0
    def load_dataset(self, root, batch_size = 1):
        self.dataloaderA = Dataloader(root = root[0], batch_size = batch_size)
        self.dataloaderB = Dataloader(root = root[1], batch_size = batch_size)
        
    def build_model(self, lr, EPOCH, loss = 'MSELoss', optim = 'Adam', betas = (0.5, 0.999), cycleLambda = 10):
        self.EPOCH = EPOCH
        self.cycleLambda = cycleLambda
        self.lr = lr
        self.optim = optim
        self.betas = betas
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.GeneratorA = Generator().cuda(self.device)
            self.GeneratorB = Generator().cuda(self.device)
            self.DiscriminatorA = Discriminator().cuda(self.device)
            self.DiscriminatorB = Discriminator().cuda(self.device)
            
        else:
            self.device = torch.device('cpu')
            self.GeneratorA = Generator()
            self.GeneratorB = Generator()
            self.DiscriminatorA = Discriminator()
            self.DiscriminatorB = Discriminator()   
        
        if 'GeneratorA.pkl' in os.listdir():
            self.GeneratorA = torch.load('GeneratorA.pkl')
        if 'DiscriminatorA.pkl' in os.listdir():
            self.DiscriminatorA = torch.load('DiscriminatorA.pkl')        
        if 'GeneratorB.pkl' in os.listdir():
            self.GeneratorB = torch.load('GeneratorB.pkl')
        if 'DiscriminatorB.pkl' in os.listdir():
            self.DiscriminatorB = torch.load('DiscriminatorB.pkl')
        
        self.criterion = utils.Loss(loss = loss)
        self.L1Loss = utils.Loss(loss = 'L1Loss')

    def train(self, Identity = True):
        #### Check build_model ####
        try:
            getattr(self, 'GeneratorA')
            getattr(self, 'GeneratorB')

            getattr(self, 'DiscriminatorA')
            getattr(self, 'DiscriminatorB')

            getattr(self, 'criterion')
            getattr(self, 'L1Loss')

        except:
            assert False, 'Not apply build_model'
        
        #### Check load_dataset ####
        try:
            getattr(self, 'dataloaderA')
            getattr(self, 'dataloaderB')

        except:
            assert False, 'Not apply load_dataset'
        
        ImagePoolA = []
        ImagePoolB = []
        
        for epoch in range(self.EPOCH):
            if epoch < self.step:
                continue
            if epoch >= 100:
                lr = 0.0002 - 0.0002*(epoch-100)/100
                
            for i, data in enumerate(zip(self.dataloaderA, self.dataloaderB)):
                real_A = data[0][0].to(self.device)
                real_B = data[1][0].to(self.device)
                
                self.optimG_A = utils.optim(self.GeneratorA, lr = self.lr, optim = self.optim, betas =self. betas)
                self.optimG_B = utils.optim(self.GeneratorB, lr = self.lr, optim = self.optim, betas = self.betas)
                self.optimD_A = utils.optim(self.DiscriminatorA, lr = self.lr, optim = self.optim, betas = self.betas)
                self.optimD_B = utils.optim(self.DiscriminatorB, lr = self.lr, optim = self.optim, betas = self.betas)
                
                fake_A = self.GeneratorA(real_B)
                fake_B = self.GeneratorB(real_A)
                
                recon_B = self.GeneratorB(fake_A)
                recon_A = self.GeneratorA(fake_B)

                DA_real_A = self.DiscriminatorA(real_A)
                DB_real_B = self.DiscriminatorB(real_B)
                
                DA_fake_A = self.DiscriminatorA(fake_A)
                DB_fake_B = self.DiscriminatorB(fake_B)
                
                
                ### train Generator ###
                self.optimG_A.zero_grad()
                self.optimG_B.zero_grad()
                
                lossG_A = self.criterion(DA_fake_A, torch.ones(DA_fake_A.shape).to(self.device))
                lossG_B = self.criterion(DB_fake_B, torch.ones(DB_fake_B.shape).to(self.device))
                
                cycleLoss= (self.L1Loss(recon_A, real_A) + self.L1Loss(recon_B, real_B)) * self.cycleLambda
                
                LossG = lossG_A + lossG_B + cycleLoss
                if Identity:
                    LossG += 0.5 * self.cycleLambda * (self.L1Loss(fake_B, real_A) + self.L1Loss(fake_A, real_B))
                LossG.backward(retain_graph = True)
                
                self.optimG_A.step()
                self.optimG_B.step()

                
                ### train Discriminator ###
                
                fake_A, p_A = ImagePool(ImagePoolA, fake_A, device = self.device)
                fake_B, p_B = ImagePool(ImagePoolB, fake_B, device = self.device)
            
                DA_real_A = self.DiscriminatorA(real_A)
                DB_real_B = self.DiscriminatorB(real_B)
                self.optimD_A.zero_grad()
                self.optimD_B.zero_grad()
                
                
                
                lossA_real = self.criterion(DA_real_A, torch.ones(DA_real_A.shape).to(self.device))
                lossA_fake = self.criterion(DA_fake_A,torch.zeros(DA_fake_A.shape).to(self.device))
                lossB_real = self.criterion(DB_real_B, torch.ones(DB_real_B.shape).to(self.device))
                lossB_fake = self.criterion(DB_fake_B, torch.zeros(DB_real_B.shape).to(self.device))
                
                (lossA_real + lossA_fake).backward(retain_graph = True)
                (lossB_real + lossB_fake).backward(retain_graph = True)
                
                self.optimD_A.step()
                self.optimD_B.step()    

                utils.PresentationExperience(epoch, i, 100, G = LossG.item(), D_A = (lossA_real+lossA_fake).item(), D_B = (lossB_real+lossB_fake).item())
                
                if i % 100 == 99:
                    torch.save(self.DiscriminatorA, 'DiscriminatorA.pkl')
                    torch.save(self.DiscriminatorB, 'DiscriminatorB.pkl')
                    torch.save(self.GeneratorA, 'GeneratorA.pkl')
                    torch.save(self.GeneratorB, 'GeneratorB.pkl')
