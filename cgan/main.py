from cgan import CGAN


lr = 0.0001
root = '/Users/mac/Documents/GitHub/GAN/google/data/'
batch_size = 128

model = CGAN()
model.load_dataset(root = root, batch_size = batch_size)
model.build_model(lr = lr, Epoch = 50)

model.train()