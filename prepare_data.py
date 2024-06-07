import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
import numpy as np

batch_size_train = 128
batch_size_valid = 1

def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST('./dataset',
                                          train=True,
                                          download=True,
                                          transform=transform)
    valideset = torchvision.datasets.MNIST('./dataset',
                                           train=False,
                                           download=True,
                                           transform=transform)

    return trainset, valideset


def get_trainloader(trainset):
    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=0)

    return trainloader


def get_validloader(validset):
    validloader = DataLoader(validset, batch_size=batch_size_valid, shuffle=True, num_workers=0)

    return validloader


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show():
    trainset, validset = get_data()
    trainloader = get_trainloader(trainset)
    validloader = get_validloader(validset)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid)
    plt.show()
