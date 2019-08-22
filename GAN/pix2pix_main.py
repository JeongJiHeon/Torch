#-*- coding: utf-8 -*-
    
    
import pix2pix
import model
import torch
import numpy as np
import utils
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
datarootA = '/root/Deeplearning/DCGAN/cityscapes/city'
datarootB = '/root/Deeplearning/DCGAN/cityscapes/scapes'

#datarootA = '/Users/mac/Documents/GitHub/GAN/data/cityscapes/real'
#datarootB = '/Users/mac/Documents/GitHub/GAN/data/cityscapes/target'


LAMBDA = 100
EPOCH = 200
batch_size = 1

Generator , Discriminator = model.model('pix2pix')
G = Generator()
D = Discriminator()

#G = torch.load('pix2pix_Generator.pkl')
#D = torch.load('pix2pix_Discriminator.pkl')
G = utils.load_model(G, name = '/root/Deeplearning/DCGAN/pix2pix_Generator.pkl')
D = utils.load_model(D, name = '/root/Deeplearning/DCGAN/pix2pix_Discriminator.pkl')
print('----Finish Load----')

criterion = torch.nn.BCELoss()
L1Loss = torch.nn.L1Loss()
optimG = torch.optim.Adam(G.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimD = torch.optim.Adam(D.parameters(), lr = 0.0002, betas = (0.5, 0.999))

datasetA = dset.ImageFolder(root = datarootA, transform =  transforms.Compose( 
            [transforms.Resize(512), 
             transforms.CenterCrop(512), 
             transforms.ToTensor(), 
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ))


datasetB = dset.ImageFolder(root = datarootB, transform =  transforms.Compose( 
            [transforms.Resize(512), 
             transforms.CenterCrop(512), 
             transforms.ToTensor(), 
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ))

dataloaderB = torch.utils.data.DataLoader(datasetB, batch_size = 1, shuffle = False, num_workers = 2)

fix_dataB = next(iter(dataloaderB))[0]
def main():
    
    img_list = []
    epoch_img_list = []
    fix_latent = torch.randn(1,3,512,512)
    
    real_label = torch.ones(batch_size,1, 16, 16)
    fake_label = torch.zeros(batch_size,1, 16,16)
    
    
    for epoch in range(EPOCH):
        dataloaderA = torch.utils.data.DataLoader(datasetA, batch_size = 1, shuffle = False, num_workers = 2)
        dataloaderB = torch.utils.data.DataLoader(datasetB, batch_size = 1, shuffle = False, num_workers = 2)



        if 'pix2pix_epoch_img_list.npy' in os.listdir():
            img_list = list(np.load('pix2pix_epoch_img_list.npy'))
        img_list.append(G(fix_dataB).detach().numpy())
        np.save('pix2pix_epoch_img_list', img_list)
        
        
        for i in range(len(dataloaderA)):
            
            dataA = next(iter(dataloaderA))[0]
            dataB = next(iter(dataloaderB))[0]
            
            fake = G(dataB)
            
            optimD.zero_grad()

            D_Loss = criterion(D(dataB, dataA), real_label) + criterion(D(fake, dataA), fake_label)
            D_Loss.backward(retain_graph=True)
            
            optimD.step()
            
            optimG.zero_grad()
            G_Loss = criterion(D(fake, dataA), real_label) + L1Loss(fake, dataA) * LAMBDA
            G_Loss.backward(retain_graph=True)
            
            optimG.step()
            
            
            if i % 5 == 0:
                print("[{}/{}]  G : {}  D : {}".format(
                    len(dataloaderA), i, G_Loss.item(), D_Loss.item()
                ))
                
                utils.save_model(G, name = 'pix2pix_Generator.pkl')
                utils.save_model(D, name = 'pix2pix_Discriminator.pkl')
            if i % 1000 == 0 :
                if 'pix2pix_imglist.npy' in os.listdir():
                    img_list = list(np.load('pix2pix_imglist.npy'))
                img_list.append(G(fix_dataB).detach().numpy())
                np.save('pix2pix_imglist', img_list)
    
    
    














