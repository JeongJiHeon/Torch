import VAE


lr = 0.0002, EPOCH = 200
root = 

vae = VAE.VAE()
vae.build_model(lr = lr, EPOCH = EPOCH)
vae.load_dataset(root = root)




vae.train()