import torch
import torch.nn as nn
import cgan
import utils
import torchvision
import model
import os
import numpy as np

        
EPOCH = 50
batch_size = 128
Generator, Discriminator = model.model('cgan')
G = Generator()
D = Discriminator()
G = utils.load_model(G, name = 'cgan_Generator.pkl')
D = utils.load_model(D, name = 'cgan_Discriminator.pkl')
mnist = utils.dataset('mnist')
criterion = nn.BCELoss()
optimG = torch.optim.Adam(G.parameters(),lr = 0.0002, betas = (0.5, 0.999))
optimD = torch.optim.Adam(D.parameters(),lr = 0.0002, betas = (0.5, 0.999))
fix_latent = torch.rand(64,100)
fix_y = torch.zeros(64, 10)
fix_y.scatter_(1, next(iter(mnist))[1][0:64].view(64, 1), 1)


def main():
    img_list = []
    for epoch in range(EPOCH):
        for i, data in enumerate(mnist):
            noise = torch.rand(batch_size,100)
            real_label = torch.ones(batch_size, 1)
            fake_label = torch.zeros(batch_size, 1)

            y_real = torch.zeros(batch_size, 10)
            y_real.scatter_(1, data[1].view(batch_size, 1), 1)


            y_fake = torch.zeros(batch_size, 10)
            y_fake.scatter_(1,(torch.rand(batch_size,1)*10).type(torch.LongTensor).view(batch_size, 1),1)                    
            real = data[0].squeeze(1).reshape([batch_size,784])
            fake = G(noise, y_fake)

            optimD.zero_grad()
            D_loss = criterion(D(real, y_real), real_label) + criterion(D(fake, y_fake),fake_label)
            D_loss.backward(retain_graph=True)
            optimD.step()


            optimG.zero_grad()
            G_loss = criterion(D(fake, y_fake), real_label)
            G_loss.backward()
            optimG.step()

            if i % 50 == 0:
                print("[{}/{}]  G : {}  D : {}".format(
                    len(mnist), i, G_loss.item(), D_loss.item()
                ))
                
                utils.save_model(G, name = 'cgan_Generator.pkl')
                utils.save_model(D, name = 'cgan_Discriminator.pkl')
            if i % 50 == 0 :
                if 'cgan_imglist.npy' in os.listdir():
                    img_list = list(np.load('cgan_imglist.npy'))
                img_list.append(G(fix_latent, fix_y).detach().numpy())
                np.save('cgan_imglist', img_list)



        