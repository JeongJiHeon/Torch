import torch
import model
import utils


dataroot = '/Users/mac/python/dataset'

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
    
    optim = model.optim(G, D, betas=(0.5, 0.999), opt = 'adam', lr = 0.002)
    criterion = model.loss('BCELoss')
    
    dataloader = utils.data_loader(dataroot, batch_size = batch_size, image_size = image_size, num_workers = workers)
    for epoch in range(EPOCH):
        for i, data in enumerate(dataloader):
            for opt in optim:
                opt.zero_grad()
            
            latent = torch.randn(batch_size, 100, 1, 1)
            
            label_real = torch.full((batch_size,), real_label)
            label_fake = torch.full((batch_size,), fake_label)
            
            err_D = criterion(D(G(latent)).view(-1), label_fake) + criterion(D(data[0]).view(-1), label_real)
            err_G = criterion(D(G(latent)).view(-1), label_real)
            
            err_D.backward(retain_graph=True)
            err_G.backward()
            
            for opt in optim:
                opt.step()
            
            if i % 100 == 0:
                print("[%d/%d]" % (i, len(dataloader)), end = ' ')
                print("G Loss : ", Loss_G.detach().numpy(), end = ' ')
                print("D Loss : ", Loss_D.detach().numpy())
                utils.save_model(G)
                utils.save_model(D)
                
                
                



    
    
    
if __name__=='__main__':
    main()