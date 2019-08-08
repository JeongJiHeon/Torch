import torch
import model
import utils
import numpy as np
import os

dataroot = '/root/Deeplearning'
#dataroot = '/Users/mac/python/dataset'

image_size = 64
batch_size = 128
workers = 2
EPOCH = 5
real_label = 1
fake_label = 0


def main():    
    
    Generator, Discriminator = model.model('dcgan')
    G = Generator()
    D = Discriminator()
    utils.load_model(G)
    utils.load_model(D)
    
    optim_G = torch.optim.Adam(G.parameters(), lr = 0.0002, betas=(0.5,0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr = 0.0002, betas=(0.5,0.999))


    criterion = model.loss('BCELoss')
    
    dataloader = utils.data_loader(dataroot, batch_size = batch_size, image_size = image_size, num_workers = workers)
    
    fix_latent = torch.randn(64, 100, 1, 1)
    label_real = torch.full((batch_size,), real_label)
    label_fake = torch.full((batch_size,), fake_label)
    img_list = []
    for epoch in range(EPOCH):
        for i, data in enumerate(dataloader):
            latent = torch.randn(batch_size, 100, 1, 1)
            optim_D.zero_grad()
            
            err_D_real = criterion(D(data[0]).view(-1), label_real)
            err_D_fake = criterion(D(G(latent)).view(-1), label_fake)
            err_D_real.backward(retain_graph = True)
            err_D_fake.backward()

            optim_D.step()
            
            optim_G.zero_grad()
            err_G = criterion(D(G(latent)).view(-1), label_real)


            err_G.backward()
            
            optim_G.step()
            
            
            if i % 50 == 0:
                print("[%d/%d]" % (i, len(dataloader)), end = ' ')
                print("real : %.4f" % err_D_real.item(), end = '   ')
                print("fake : %.4f" % err_D_fake.item(), end = '   ')
                print("G : %.4f" % err_G.item())
                
                utils.save_model(G)
                utils.save_model(D)
            if i % 500 == 0 :
                if 'imglist.npy' in os.listdir():
                    img_list = list(np.load('imglist.npy'))
                img_list.append(G(fix_latent).detach().numpy())
                np.save('imglist', img_list)
                    
                
                

    
if __name__=='__main__':
    main()