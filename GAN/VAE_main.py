import model
import utils
import torch
import torchvision
import os
import numpy as np



batch_size = 128
EPOCH = 30
lr = 0.0005



mnist = utils.dataset('mnist')
VAE = model.model('VAE')


vae = VAE()
vae = utils.load_model(vae, name = 'vae.pkl')

optim = torch.optim.Adam(vae.parameters(),lr = lr)

fix_data = next(iter(mnist))[0].reshape(128,784)[0:64]

        
def main():
    img_list = []
    
    for epoch in range(EPOCH):
        for i, data in enumerate(mnist):
            
            optim.zero_grad()
            x_hat, output, loss, mll, KL= vae(data[0].reshape(128,784))
            loss.backward()
            optim.step()


            if i % 50 == 0:
                print("[{}/{}][{}/{}] loss:{}, marginal_loss:{}, KL:{}".format(epoch+1, EPOCH, i, len(mnist), loss.item(), mll.item(), KL.item()))

                utils.save_model(vae, name = 'vae.pkl')
                if 'vae_imglist.npy' in os.listdir():
                    img_list = list(np.load('vae_imglist.npy'))
                img_list.append(vae(fix_data)[1].reshape(64,28,28).detach().numpy())
                np.save('vae_imglist', img_list)



