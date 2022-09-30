import argparse

import torch
import torchvision
import numpy
import time
from torch import nn
# import matplotlib
# import matplotlib.pyplot as plt
from torch.autograd import Variable
from logConfig import logConfig
from model.resnet20 import ResNet20
# from transformer_resnet import ResNet20
from dataset_prepare.cifar10 import read_cifar10
import os
os.environ['CUDA_VISIBLE_DEVICES']='1,2'
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms as transforms
from datetime import datetime

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
def deparallel_train():
    since = time.time()
    data_dir = './dataset/cifar'
    model_dir = './cnn'
    log_dir = './log'
    batchsize = 128
    n_epochs = 200
    best_acc = 0.0  # 记录测试最高准确率
    Lr = 0.1

    data_loader_train, data_loader_test = read_cifar10(batchsize, data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet20().to(device)
    print(model)
    # 取出权重参数和偏置参数，仅对权重参数加惩罚系数
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    cost = nn.CrossEntropyLoss().to(device)  # Q3：要不要加正则项

    # L1正则
    # regularization_loss = 0
    # for param in model.parameters():
    #     regularization_loss += torch.sum(torch.abs(param))
    #
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay': 1e-4},
                                 {'params': bias_p, 'weight_decay': 0}], lr=Lr,
                                momentum=0.9)  # 内置的SGD是L2正则，且对所有参数添加惩罚，对偏置加正则易导致欠拟合，一般只对权重正则化
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 122, 163], gamma=0.1,
                                                     last_epoch=-1)  # Q4: 对应多少步,epoch= 32000/(50000/batch_size),48000,64000

    Loss_list = []
    Accuracy_list = []
    for epoch in range(n_epochs):

        model.train()

        training_loss = 0.0
        training_correct = 0
        training_acc = 0.0
        print("Epoch {}/{}".format(epoch + 1, n_epochs))
        print("-" * 30)

        total_train = 0
        for i, data in enumerate(data_loader_train):
            x, labels = data
            x, labels = x.to(device), labels.to(device)

            # print(x.shape)
            # print(label.shape)
            # 前向传播计算损失
            outputs = model(x)
            loss = cost(outputs, labels)
            training_loss += loss.item()
            # print(outputs)
            _, pred = torch.max(outputs, 1)  # 预测最大值所在位置标签
            total_train += labels.size(0)
            num_correct = (pred == labels).sum()
            training_acc += num_correct.item()

            # 反向传播+优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('Epoch：', epoch, 'train loss:', training_loss/len(data_loader_train))
            # Loss_list.append(training_loss/len(data_loader_train))
            if i % 100 == 99:
                print('[%d, %5d] traing_loss: %f' % (epoch + 1, i + 1, training_loss / 100))
                logger.info('[%d, %5d] traing_loss: %f' % (epoch + 1, i + 1, training_loss / 100))
                Loss_list.append(training_loss / 100)
                training_loss = 0.0
        print('Train acc:{:.4}%'.format(100 * training_acc / total_train))
        logger.info('Train acc:{:.4}%'.format(100 * training_acc / total_train))

        scheduler.step()

        model.eval()
        testing_correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader_test:
                x_test, label_test = data
                x_test, label_test = x_test.to(device), label_test.to(device)
                outputs = model(x_test)
                _, pred = torch.max(outputs.data, 1)
                total += label_test.size(0)
                testing_correct += (pred == label_test).sum().item()
        print('Test acc: {:.4}%'.format(100 * testing_correct / total))
        logger.info('Test acc: {:.4}%'.format(100 * testing_correct / total))
        Accuracy_list.append(100 * testing_correct / total)
        acc = 100 * testing_correct / total
        if acc > best_acc:
            best_acc = acc
            best_acc_loc = epoch
        # print("Loss :{:.4f}, Train acc :{.4f}, Test acc :{.4f}".format(training_loss/len(data_train),100*training_correct/len(data_train),100*testing_correct/len(data_test)))

    print('test best acc:{}% at epoch{}'.format(best_acc, best_acc_loc))
    logger.info('test best acc:{}% at epoch{}'.format(best_acc, best_acc_loc))
    time_used = time.time() - since
    print('-' * 30)
    print('训练用时： {:.0f}m {:.0f}s'.format(time_used // 60, time_used % 60))
    logger.info('训练用时： {:.0f}m {:.0f}s'.format(time_used // 60, time_used % 60))
    # print('最高准确率: {}%'.format(100 * best_acc))
    logger.info('最高准确率: {}%'.format(100 * best_acc))

    # 保存参数优化器等
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, os.path.join(model_dir, '{}best_acc.pth'.format(best_acc)))

def parallel_train(gpu,args):
    ############################################################
    rank = args.nr * args.gpus + gpu
    dist.init_process_group( #初始化分布式环境
        backend='nccl',   #各个进程的通信协议
        init_method='env://', #参数化方法
        world_size=args.world_size, #环境变量的配置
        rank=rank
    )
    ############################################################
    #这里设置了种子，定义了模型，设置了gpu，数据集块，交叉熵loss，SGD优化器
    torch.manual_seed(0)
    model = ResNet20()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    ###############################################################
    # Wrap the model 通过DistributedDataParallel包装模型
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    ###############################################################

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
        root='./dataset/mnist',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    ################################################################
    #通过DistributedSampler采样数据
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    ################################################################

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        ##############################
        shuffle=False,  #
        ##############################
        num_workers=0,
        pin_memory=True,
        #############################
        sampler=train_sampler)  #
    #############################

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    total_step,
                    loss.item())
                   )
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    total_step,
                    loss.item())
                   )
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
        logger.info("Training complete in: " + str(datetime.now() - start))

def main():
    #导入argparse方便使用sh脚本传值
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N') #节点数
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes') #node的序号
    parser.add_argument('--epochs', default=5, type=int,
                        metavar='N',
                        help='number of total epochs to run')  #总训练轮数
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes  #
    os.environ['MASTER_ADDR'] = 'localhost'  #主进程的IP
    os.environ['MASTER_PORT'] = '8888'  #
    if args.world_size == 1:
        deparallel_train2(args)
    else:
        mp.spawn(parallel_train, nprocs=args.gpus, args=(args,))  #
    #########################################################

def deparallel_train2(args,gpu=0):
    #pytorch是cpu版的
    logger = logConfig("deparallel_train2.log").logger
    rank = 0
    #这里设置了种子，定义了模型，设置了gpu，数据集块，交叉熵loss，SGD优化器
    torch.manual_seed(0)
    model = ResNet20()
    # torch.cuda.set_device(gpu)
    # model.cuda(gpu)
    batch_size = 300
    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet20().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Data loading code
    train_dataset = torchvision.datasets.CIFAR10(
        root='./dataset/cifar',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    ################################################################

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        ##############################
        shuffle=False,  #
        ##############################
        num_workers=0,
        pin_memory=True,
        #############################
        )  #
    #############################

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 :
                logger.info('rank [{}],Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    rank,
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    total_step,
                    loss.item())
                   )
    if rank == 0:
        print("Training complete in: " + str(datetime.now() - start))
        logger.info("rank [{}],Training complete in: ".format(rank) + str(datetime.now() - start))

if __name__ == '__main__':

    main()