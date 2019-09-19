#-*- coding: utf-8 -*-
    
import cyclegan    
import model
import torch
import numpy as np
import utils
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
datarootA = '/root/deeplearning/ukiyoe2photo/trainA'
datarootB = '/root/deeplearning/ukiyoe2photo/trainB'

#datarootA = '/Users/mac/Documents/GitHub/GAN/data/ukiyoe2photo/trainA'
#datarootB = '/Users/mac/Documents/GitHub/GAN/data/ukiyoe2photo/trainB'


LAMBDA = 100
EPOCH = 200
batch_size = 1
learning_rate = 0.0002

Generator , Discriminator = model.model('cyclegan')
G_A = Generator()
G_B = Generator()

D_A = Discriminator()
D_B = Discriminator()

#G = torch.load('pix2pix_Generator.pkl')
#D = torch.load('pix2pix_Discriminator.pkl')

'''
====================================
G_A = A를 만드는 Generator (input = B data, output = A data)
G_B = B를 만드는 Generator (input = A data, output = B data)

D_A = A 인지 확인하는 Discriminator (input = A data)
D_B = B 인지 확인하는 Discriminator (input = B data)
====================================
'''

G_A = utils.load_model(G_A, name = 'cyclegan_Generator_A.pkl')
D_A = utils.load_model(D_A, name = 'cyclegan_Discriminator_A.pkl')

G_B = utils.load_model(G_B, name = 'cyclegan_Generator_B.pkl')
D_B = utils.load_model(D_B, name = 'cyclegan_Discriminator_B.pkl')




datasetA = dset.ImageFolder(root = datarootA, transform =  transforms.Compose( 
            [transforms.Resize(256), 
             transforms.CenterCrop(256), 
             transforms.ToTensor(), 
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ))


datasetB = dset.ImageFolder(root = datarootB, transform =  transforms.Compose( 
            [transforms.Resize(256), 
             transforms.CenterCrop(256), 
             transforms.ToTensor(), 
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ))

dataloaderB = torch.utils.data.DataLoader(datasetB, batch_size = 1, shuffle = True, num_workers = 0)
dataloaderA = torch.utils.data.DataLoader(datasetA, batch_size = 1, shuffle = True, num_workers = 0)

img_list = []

if 'cyclegan_fix_data_A.npy' in os.listdir():
    fix_dataA = torch.Tensor(np.load('cyclegan_fix_data_A.npy'))
    fix_dataB = torch.Tensor(np.load('cyclegan_fix_data_B.npy'))
else:
    fix_dataA = (next(iter(dataloaderA))[0] +1)/2
    fix_dataB = (next(iter(dataloaderB))[0] +1)/2
    np.save('cyclegan_fix_data_A.npy', fix_dataA.detach().numpy())
    np.save('cyclegan_fix_data_B.npy', fix_dataB.detach().numpy())
    img_list.append(torch.cat([G_A(fix_dataB).detach(), G_B(fix_dataA).detach()]).numpy())
    np.save('cyclegan_epoch_img_list', img_list)

real_label = torch.ones(batch_size, 512, 16, 16)
fake_label = torch.zeros(batch_size, 512, 16, 16)

criterion = torch.nn.BCELoss()
L1Loss = torch.nn.L1Loss()
cycleLambda = 10

print('----Finish Load----')


def main():
    img_list = []

    
    lr = learning_rate
    for epoch in range(EPOCH):
        
        if epoch >= 50:
            lr *= (epoch-1)
            lr /= (epoch)


        optimG_A = torch.optim.Adam(G_A.parameters(), lr = lr, betas = (0.9, 0.99))
        optimD_A = torch.optim.Adam(D_A.parameters(), lr = lr, betas = (0.9, 0.99))

        optimG_B = torch.optim.Adam(G_B.parameters(), lr = lr, betas = (0.9, 0.99))
        optimD_B = torch.optim.Adam(D_B.parameters(), lr = lr, betas = (0.9, 0.99))
        

        if 'cyclegan_epoch_img_list.npy' in os.listdir():
            img_list = list(np.load('cyclegan_epoch_img_list.npy'))

        img_list.append(torch.cat([G_A(fix_dataB).detach(), G_B(fix_dataA).detach()]).numpy())
        np.save('cyclegan_epoch_img_list', img_list)
        
        
        for i, data in enumerate(zip(iter(dataloaderA), iter(dataloaderB))):
            dataA = (data[0][0] +1)/2
            dataB = (data[1][0] +1)/2
            
            optimD_A.zero_grad()
            optimD_B.zero_grad()
            
            fake_A = G_A(dataB) # Make A used by Generator_A
            fake_B = G_B(dataA) # Make B used by Generator_B
            
            cycle_A = G_A(fake_B) # Make cycleA used by Generator_A in fake_B
            cycle_B = G_B(fake_A) # Make cycleB used by Generator_B in fake_A
            

            
            
            Loss_Da_G_X_Y = criterion(D_A(dataA), real_label) + criterion(D_A(fake_A), fake_label)
            Loss_Da_G_X_Y.backward(retain_graph=True)
            
            Loss_Db_F_Y_X = criterion(D_B(dataB), real_label) + criterion(D_B(fake_B), fake_label)
            Loss_Db_F_Y_X.backward(retain_graph=True)
            
            optimD_A.step()
            optimD_B.step()
            print("[{}/{}]".format(i, len(dataloaderA)), end = ' ')
            print("Discriminator Finish", end = ' ')
            
            optimG_A.zero_grad()
            optimG_B.zero_grad()
            
            Loss_G_A = criterion(D_A(fake_A), real_label)
            Loss_G_A.backward(retain_graph=True)
            
            Loss_G_B = criterion(D_B(fake_B), real_label)
            Loss_G_B.backward(retain_graph=True)
            
            cycleLoss_G_F = cycleLambda * (L1Loss(cycle_A, dataA) + L1Loss(cycle_B, dataB))
            cycleLoss_G_F.backward(retain_graph=True)
            
            optimG_A.step()
            optimG_B.step()
            
            print("Generator Finish")


            
            print("D_A : {}, D_B : {}, G_A : {}, G_B : {}, cycleLoss : {}".format(Loss_Da_G_X_Y.item(), Loss_Db_F_Y_X.item(), 
                                                                Loss_G_A.item(), Loss_G_B.item(), cycleLoss_G_F.item()))
            
            if i % 100 == 0:
                utils.save_model(G_A, name = 'cyclegan_Generator_A.pkl')
                utils.save_model(D_A, name = 'cyclegan_Discriminator_A.pkl')

                utils.save_model(G_B, name = 'cyclegan_Generator_B.pkl')
                utils.save_model(D_B, name = 'cyclegan_Discriminator_B.pkl')


            

