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
datarootA = '/root/deeplearning/monet2photo/trainA'
datarootB = '/root/deeplearning/monet2photo/trainB'

#datarootA = '/root/deeplearning/ukiyoe2photo/trainA'
#datarootB = '/root/deeplearning/ukiyoe2photo/trainB'

#datarootA = '/Users/mac/Documents/GitHub/GAN/data/ukiyoe2photo/trainA'
#datarootB = '/Users/mac/Documents/GitHub/GAN/data/ukiyoe2photo/trainB'


cycleLambda = 10
EPOCH = 200
batch_size = 1
learning_rate = 0.0002

Generator , Discriminator = model.model('cyclegan')
G_A = Generator() # input = B Make A image
G_B = Generator() # input = A Make  B image

D_A = Discriminator() # input = A
D_B = Discriminator() # input = B

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

dataloaderB = torch.utils.data.DataLoader(datasetB, batch_size = 1, shuffle = True, num_workers = 2)
dataloaderA = torch.utils.data.DataLoader(datasetA, batch_size = 1, shuffle = True, num_workers = 2)

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

real_label = torch.ones(D_A(fix_dataB).shape)
fake_label = torch.zeros(D_A(fix_dataB).shape)

criterion = torch.nn.MSELoss()
L1Loss = torch.nn.L1Loss()


    
print('----Finish Load----')


def main():
    img_list = []
    
    if 'trainepoch.npy' in os.listdir():
        trainepoch = int(np.load('trainepoch.npy'))
    else:
        trainepoch = 0
    
    if 'ImagePool_A.npy' in os.listdir():
        ImagePool_A = list(np.load('ImagePool_A.npy'))
        ImagePool_B = list(np.load('ImagePool_B.npy'))
    else:
        ImagePool_A = []
        ImagePool_B = []
    
    
    
    lr = learning_rate
    for epoch in range(EPOCH):
        
        if trainepoch >= epoch:
            continue
        
        if epoch >= 100:
            lr = 0.0002 - 0.0002*(epoch-100)/100



        optimG_A = torch.optim.Adam(G_A.parameters(), lr = lr, betas = (0.5, 0.999))
        optimD_A = torch.optim.Adam(D_A.parameters(), lr = lr, betas = (0.5, 0.999))

        optimG_B = torch.optim.Adam(G_B.parameters(), lr = lr, betas = (0.5, 0.999))
        optimD_B = torch.optim.Adam(D_B.parameters(), lr = lr, betas = (0.5, 0.999))
        

        if 'cyclegan_epoch_img_list.npy' in os.listdir():
            img_list = list(np.load('cyclegan_epoch_img_list.npy'))

        img_list.append(torch.cat([G_A(fix_dataB).detach(), G_B(fix_dataA).detach()]).numpy())
        np.save('cyclegan_epoch_img_list', img_list)
        
        
        for i, data in enumerate(zip(iter(dataloaderA), iter(dataloaderB))):
            dataA = (data[0][0])
            dataB = (data[1][0])
            
            fake_A = G_A(dataB) # Make A used by Generator_A
            fake_B = G_B(dataA) # Make B used by Generator_B
            cycle_A = G_A(fake_B) # Make cycleA used by Generator_A in fake_B
            cycle_B = G_B(fake_A) # Make cycleB used by Generator_B in fake_A
            
   ######## ---- Train Generator ---- ########



            optimG_A.zero_grad()
            optimG_B.zero_grad()
        
            D_A_fakeA = D_A(fake_A)
            D_B_fakeB = D_B(fake_B)            
            
            Loss_G_A = criterion(D_A_fakeA, real_label) 
            Loss_G_B = criterion(D_B_fakeB, real_label)
            
            Identity_Loss_G_A = L1Loss(G_A(dataA), dataA) + 0.5 * cycleLambda
            Identity_Loss_G_B = L1Loss(G_B(dataB), dataB) + 0.5 * cycleLambda

            
            cycleLoss_G_F = cycleLambda * (L1Loss(cycle_A, dataA) + L1Loss(cycle_B, dataB))
            
            LossG = Loss_G_A + Loss_G_B + cycleLoss_G_F + Identity_Loss_G_A + Identity_Loss_G_B
            LossG.backward(retain_graph=True)
                        
            optimG_A.step()
            optimG_B.step()
            
            
   ######## ---- Train Discriminator ---- ########
            
            optimD_A.zero_grad()
            optimD_B.zero_grad()
            
            fake_A, p_A = utils.ImagePool(ImagePool_A, fake_A)
            fake_B, p_B = utils.ImagePool(ImagePool_B, fake_B)
            
            D_A_dataA = D_A(dataA)
            D_B_dataB = D_B(dataB)
            
            if p_A:
                D_A_fakeA = D_A(fake_A)
            if p_B:
                D_B_fakeB = D_B(fake_B)


            Loss_Da_G_X_Y = (criterion(D_A_dataA, real_label) + criterion(D_A_fakeA, fake_label))/2
            Loss_Da_G_X_Y.backward(retain_graph=True)
            
            Loss_Db_F_Y_X = (criterion(D_B_dataB, real_label) + criterion(D_B_fakeB, fake_label))/2
            Loss_Db_F_Y_X.backward(retain_graph=True)
            
            optimD_A.step()
            optimD_B.step()
            np.save('ImagePool_A', ImagePool_A)
            np.save('ImagePool_B', ImagePool_B)
            print("[{}] [{}/{}] D_A : {:.4f}, D_B : {:.4f}, G_A : {:.4f}, G_B : {:.4f}, cycleLoss : {:.4f}".format(epoch, i, len(dataloaderA), 
                                                                                                                   Loss_Da_G_X_Y.item(), Loss_Db_F_Y_X.item(), 
                                                                (Loss_G_A+Identity_Loss_G_A).item(), (Loss_G_B+Identity_Loss_G_B).item(), cycleLoss_G_F.item()))
            
            if i % 100 == 0:
                utils.save_model(G_A, name = 'cyclegan_Generator_A.pkl')
                utils.save_model(D_A, name = 'cyclegan_Discriminator_A.pkl')

                utils.save_model(G_B, name = 'cyclegan_Generator_B.pkl')
                utils.save_model(D_B, name = 'cyclegan_Discriminator_B.pkl')
        np.save('trainepoch', np.array(epoch))


            
