import torch
import torchvision
import torch.nn as nn
import layer
import torchvision.transforms as transforms
import utils
import os

from network import *
from DataLoader import *


class CGAN(object):
    def __init__(self):
        self.step = 0
        
    def load_dataset(self, root, batch_size = 64, data = 'mnist'):
        self.dataloader = Public_Dataloader(root = root, batch_size = batch_size, data = data, t = transforms.Compose([transforms.Resize(28), transforms.ToTensor()]))
    
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
        self.criterion = utils.Loss(loss = loss)
        
        #### Build optimizing ####
        self.optimG = utils.optim(self.Generator, lr = lr, optim = optim, betas = betas)
        self.optimD = utils.optim(self.Discriminator, lr = lr, optim = optim, betas = betas)
                 
                 
    def train(self):
        #### Check build_model ####
        try:
            getattr(self, 'Generator')
            getattr(self, 'Discriminator')
            getattr(self, 'criterion')
            getattr(self, 'optimG')
            getattr(self, 'optimD')
        except:
            assert False, 'Not apply build_model'
        
        #### Check load_dataset ####
        try:
            getattr(self, 'dataloader')
        except:
            assert False, 'Not apply load_dataset'
            
            
        #### Train ####    
        for epoch in range(self.Epoch):
            if epoch < self.step:
                continue
            for i, data in enumerate(self.dataloader):

                                
                                       
                #### train Discriminator ####
                self.optimD.zero_grad()
                real_data = data[0].view(data[0].size(0), -1).to(device = self.device)
                real_label = torch.zeros(data[1].size(0), 10).to(device = self.device)
                real_label.scatter_(1, data[1].view(data[1].size(0),1).type(torch.LongTensor), 1)

                fake_data = torch.randn(data[0].size(0), 100).to(device = self.device)
                fake_label = torch.zeros(data[1].size(0), 10).to(device = self.device)
                fake_label.scatter_(1, torch.randint(0, 10, (data[1].size(0),1)).view(real_label.size(0),1).type(torch.LongTensor), 1)        

                
                fake = self.Generator(fake_data, fake_label)

                real_D = self.Discriminator(real_data, real_label)
                print(fake_data.shape)
                print(fake_label.shape)
                fake_D = self.Discriminator(fake, fake_label)
                      
                lossD = self.criterion(real_D, torch.ones(real_D.shape).to(self.device)) + self.criterion(fake_D, torch.zeros(fake_D.shape).to(self.device))
                lossD.backward(retain_graph = True)
                self.optimD.step()
                
                #### train Generator ####
                self.optimG.zero_grad()
                lossG = self.criterion(fake_D, torch.ones(fake_D.shape).to(self.device))
                lossG.backward()
                self.optimG.step()
                
                self.step += 1
                
                utils.PresentationExperience(epoch, i, 100, lossG = lossG.item(), lossD = lossD.item())
                if i % 100 == 0:
                    torch.save(self.G, 'Generator.pkl')
                    torch.save(self.D, 'Discriminator.pkl')
                    
                


                
                
        
        

        
            
            
            
            
            
            
            
            
            
            
            
            
            
            