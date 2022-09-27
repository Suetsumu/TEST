import numpy as np
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import dataset

cf10_data = torchvision.datasets.CIFAR10('dataset/cifar/', download=True)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.name = 'Net'

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def create_net(num_classes, dnn='resnet50', **kwargs):
    if dnn == 'resnet50':
        net = torchvision.models.resnet50(num_classes=num_classes)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # a1 = np.array([[1, 2, 2], [3, 4, 4]])
    # a2 = np.array([[5, 6, 6], [7, 8, 8]])
    # x = torch.tensor([a1, a2])
    #
    # # x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    # y = x.pow(3) + 0.1 * torch.randn(x.size())
    #
    # x, y = (Variable(x), Variable(y))
    dataiter = iter(dataset.trainloader)
    x,y = dataiter.next()
    # net = Net()
    # net = torchvision.models.resnet50(num_classes=num_classes)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1) #LR 逻辑回归#
    loss_func = torch.nn.MSELoss()

    plt.ion()
    plt.show()

    for t in range(5000):
        prediction = net(x)
        loss = loss_func(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         if t%5 ==0:
#             plt.cla()
#             plt.scatter(x.data.numpy(), y.data.numpy())
#             plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
#             plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
#             plt.pause(0.05)
#
# plt.ioff()
# plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
