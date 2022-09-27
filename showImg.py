import matplotlib.pyplot as plt
import numpy as np
import dataset
import torchvision
# 显示图像函数


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随机获取一些训练图像
# 图片的张量的意义，假设20*20的图片，第一个20*20矩阵存储第一张图片的R值，然后是G值，最后是B值。
dataiter = iter(dataset.trainloader)
images, labels = dataiter.next()
print(images)

# 显示图片
# imshow(torchvision.utils.make_grid(images))
# 打印标签

