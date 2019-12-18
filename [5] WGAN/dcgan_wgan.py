import torch
import torchvision
import torch.nn as nn
import layer
import torchvision.transforms as transforms
import utils
import os
import torch.autograd as autograd


from network import *
from DataLoader import *

class DCGAN(object):
    def __init__(self):
        self.step = 0
        pass
    def load_dataset(self, root, batch_size = 64, data = 'STL10', download = False):
        self.root = root
        self.dataloader = Public_Dataloader(root = root, batch_size = batch_size, data = data, download = download)
    
    def build_model(self, lr, Epoch, loss = 'BCELoss', optim = 'Adam', betas = (0, 0.9)):
        
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
             
          
    def calc_gradient_penalty(self, D, real_data, fake_data, LAMBDA = 10):
        BATCH_SIZE = real_data.shape[0]

        real_data = autograd.Variable(real_data)
        fake_data = autograd.Variable(fake_data)
        # print "real_data: ", real_data.size(), fake_data.size()
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, real_data.nelement() // BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 64, 64)
        alpha = alpha.to(device = self.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(self.device)

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device = self.device),create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty
                 
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
        
        self.fix_latent = torch.randn(64, 100, 1, 1).to(device = self.device)

        one = torch.FloatTensor([1]).cuda().mean()
        mone = torch.FloatTensor([-1]).cuda().mean()


        #### Train ####    
        for epoch in range(self.Epoch):

          
            if epoch % 1 == 0:
	            utils.saveimage(self.Generator(self.fix_latent).cpu().detach().numpy(), epoch)
            #if epoch < self.step:
            #    continue
            for i, data in enumerate(self.dataloader):
                latent = torch.randn(data[0].size(0), 100, 1, 1).to(device = self.device)
                                
                #### train Discriminator ####
                
                for p in self.Discriminator.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update
                for p in self.Generator.parameters():
                    p.requires_grad = False
                    
                self.optimD.zero_grad()
                real = data[0].to(self.device)
                fake = self.Generator(latent)
                real_D = self.Discriminator(real)
                fake_D = self.Discriminator(fake)
                      
                
                loss_real = real_D.mean()
                loss_real.backward(mone)
                
                
                loss_fake = fake_D.mean()
                loss_fake.backward(one)

                
                
                gradient_penalty = self.calc_gradient_penalty(self.Discriminator,
                                                         real, fake)
                gradient_penalty.backward()                
                lossD = loss_fake - loss_real + gradient_penalty

                self.optimD.step()
                
                #### train Generator ####
                for p in self.Discriminator.parameters():  # reset requires_grad
                    p.requires_grad = False  # they are set to False below in netG update
                for p in self.Generator.parameters():
                    p.requires_grad = True
                    
                latent = torch.randn(data[0].size(0), 100, 1, 1).to(device = self.device)
                fake = self.Generator(latent)
                fake_D = self.Discriminator(fake)
                
                self.optimG.zero_grad()
                lossG = fake_D.mean()
                lossG.backward(mone)
                self.optimG.step()
                
                self.step += 1
                
                utils.PresentationExperience(epoch, i, 100, lossG = -lossG.item(), 
                                             lossD = lossD.item())
            if epoch % 20 == 19:
                torch.save(self.Generator, 'Generator.pkl')
                torch.save(self.Discriminator, 'Discriminator.pkl')
          
            