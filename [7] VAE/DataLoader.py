import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms




def Dataloader(root, batch_size, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(64)]), shuffle = True, num_workers = 1):
    dataset = dset.ImageFolder(root = root, transform = transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)

    return dataloader
        
def Public_Dataloader(root, batch_size, t = transforms.Compose([transforms.Resize(64), transforms.ToTensor()]), 
                      shuffle = True, num_workers = 1, data = 'cifar10'):
    if data == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=t, download=True)
        
    elif data == 'mnist':
        dataset = torchvision.datasets.MNIST(root = root, train = True, transform = t, download = True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)

    return dataloader


