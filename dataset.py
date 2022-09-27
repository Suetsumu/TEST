import torchvision
import torchvision.transforms as transforms
import torch

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])]
) #预处理工具#
cf10_data = torchvision.datasets.CIFAR10('dataset/cifar/',train=True, download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(cf10_data, batch_size=2,shuffle = True,num_workers = 0)
#batch_size就是数据的个数，而不是聚类的个数#

classes = (
    'plane',
    'car',
    'bird',
    'cat',

    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck')
