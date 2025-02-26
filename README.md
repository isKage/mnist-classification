# MNIST 手写数字集分类问题：基于 AlexNet 神经网络

`Python` `PyTorch` `MNIST` `Dataset` `Dataloarder` `Tensorboard`

相关：如何读取 MNIST 数据集，搭建 AlexNet 简单卷积神经网络，模型训练和验证。

进入空目录，使用 `git` 下载
```bash
git clone https://github.com/isKage/mnist-classification.git
```

- PyTorch 的安装和环境配置可见 [zhihu](https://zhuanlan.zhihu.com/p/22230632892)
- 安装指定依赖：【进入 `requirements.txt` 根目录下安装】

```bash
pip install -r requirements.txt
```

## 0 本地配置 config.py
在根目录下创建 `config.py` 文件写入本地配置。
```python
import os
import torch
import warnings
from datetime import datetime


class DefaultConfig:
    model = 'Classification10Class'
    root = '<路径>/AllData/datasets/hojjatk/mnist-dataset'
    logdir = './logs'
    
    # 获取最新模型参数
    param_path = './checkpoints/'
    if not os.listdir(param_path):
        load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    else:
        load_model_path = os.path.join(
            param_path,
            sorted(
                os.listdir(param_path),
                key=lambda x: datetime.strptime(
                    x.split('_')[-1].split('.pth')[0],
                    "%Y-%m-%d%H%M%S"
                )
            )[-1]
        )

    lr = 0.03
    max_epochs = 1  # 暂时不调参，只训练一次
    batch_size = 64
    num_workers = 0

    print_feq = 100  # 输出频率

    if torch.cuda.is_available():
        gpu = True
        device = torch.device('cuda')
    else:
        gpu = False
        device = torch.device('cpu')

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config 参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

        config.device = torch.device('cuda:0') if config.gpu else torch.device('cpu')

        print('User config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


config = DefaultConfig()
```

## 1 读取 MNIST 数据集
直接使用 `torchvision.datasets.MNIST` 会出现网络问题，难以下载。
可以先前往 [kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) 下载。
使用 kaggle 命令下载教程可见 [从 Kaggle 下载数据集（mac 和 win 端）](https://zhuanlan.zhihu.com/p/25732245405)。

然后自定义 `get_data.py` 的 `getData` 函数读取数据集。其中 `config` 为本地配置（包含了一些参数和文件路径）。
```python
from config import config

import torchvision.datasets
from torch.utils.data import DataLoader


def getData(root=config.root, batch_size=config.batch_size):
    # 1. 准备数据集
    train_dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )

    test_dataset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )

    # 2. 获取数据集长度
    train_data_size = len(train_dataset)
    test_data_size = len(test_dataset)
    print("训练数据集长度为 {}".format(train_data_size))
    print("测试数据集长度为 {}".format(test_data_size))

    # 3. 利用DataLoader加载数据集
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader


if __name__ == "__main__":
    train_dataset, test_dataset, train_dataloader, test_dataloader = getData()
    img, label = train_dataset[0]
    print(img.shape)
    print(label)
```

## 2 搭建网络
MNIST 数据集较为简单，使用简单的 AlexNet 卷积神经网络即可
```python
import time
import torch

from torch import nn


class BasicModule(nn.Module):
    """
    作为基类，继承 nn.Module 但增加了模型保存和加载功能 save and load
    """

    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))

    def load(self, model_path):
        """
        根据模型路径加载模型
        :param model_path: 模型路径
        :return: 模型
        """
        self.load_state_dict(torch.load(model_path))

    def save(self, filename=None):
        """
        保存模型，默认使用 "模型名字 + 时间" 作为文件名，也可以自定义
        """
        if filename is None:
            filename = 'checkpoints/' + self.model_name + '_' + time.strftime("%Y-%m-%d%H%M%S") + '.pth'
        torch.save(self.state_dict(), filename)
        return filename


class Classification10Class(BasicModule):
    def __init__(self):
        super(Classification10Class, self).__init__()
        self.model_name = 'Classification10Class'
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 3 * 3, out_features=64),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x = self.module(x)
        return x


# 验证网络正确性
if __name__ == '__main__':
    classification = Classification10Class()
    # 按照batch_size=64，channel=1，size=28 * 28输入
    inputs = torch.ones((64, 1, 28, 28))
    outputs = classification(inputs)
    print(outputs.shape)
```

## 3 主程序
主程序 `main.py` 包含了训练、验证和写入 tensorboard 可视化。
```python
import models
from config import config
from get_data import getData

import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def train(**kwargs):
    config._parse(kwargs)
    classification = getattr(models, config.model)()
    classification.to(config.device)

    train_dataset, test_dataset, train_dataloader, test_dataloader = getData()
    test_data_size = len(test_dataset)

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 优化器
    learning_rate = 0.01
    optimizer = torch.optim.SGD(
        params=classification.parameters(),
        lr=config.lr,
    )

    # 7. 设置训练网络的参数
    total_train_step = 0  # 训练次数
    total_test_step = 0  # 测试次数 == epoch
    epochs = config.max_epochs  # 训练迭代次数

    # 添加tensorboard可视化
    writer = SummaryWriter("./logs")

    # 8. 开始训练
    for epoch in range(epochs):
        print("------------- 第 {} 轮训练开始 -------------".format(epoch + 1))

        # 训练步骤
        classification.train()
        for data in train_dataloader:
            # 输入输出
            images, targets = data
            images, targets = images.to(config.device), targets.to(config.device)

            outputs = classification(images)

            # 损失函数
            loss = loss_fn(outputs, targets)

            # 清零梯度
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            total_train_step += 1
            if total_train_step % config.print_feq == 0:
                print("训练次数: {}, loss: {}".format(total_train_step, loss.item()))
                writer.add_scalar(
                    tag="train_loss (every 100 steps)",
                    scalar_value=loss.item(),
                    global_step=total_train_step,
                )

        # 测试步骤(不更新参数)
        classification.eval()
        total_test_loss = 0  # 测试集损失累积
        total_accuracy = 0  # 分类问题正确率
        with torch.no_grad():
            for data in test_dataloader:
                images, targets = data
                images, targets = images.to(config.device), targets.to(config.device)

                outputs = classification(images)

                loss = loss_fn(outputs, targets)

                total_test_loss += loss.item()

                # 正确率
                accuracy = (outputs.argmax(axis=1) == targets).sum()
                total_accuracy += accuracy

        # 在测试集上的损失
        print("##### 在测试集上的 loss: {} #####".format(total_test_loss))
        writer.add_scalar(
            tag="test_loss (every epoch)",
            scalar_value=total_test_loss,
            global_step=epoch,
        )

        # 在测试集上的正确率
        print("##### 在测试集上的正确率: {} #####".format(total_accuracy / test_data_size))
        writer.add_scalar(
            tag="test_accuracy (every epoch)",
            scalar_value=total_accuracy / test_data_size,
            global_step=epoch,
        )

        # 保存每次训练的模型
        classification.save()  # 保存
        print("##### 模型成功保存 #####")

    writer.close()


if __name__ == '__main__':
    import fire

    fire.Fire()
```

## 4 运行程序
使用 `fire` 包，从而实现终端训练。
```bash
python main.py train
```
即可运行主程序的 `train` 函数。

## 5 友链

1. 关注我的知乎账号 [Zhuhu](https://www.zhihu.com/people/--55-97-8-41) 不错过我的笔记更新。
2. 我会在个人博客 [isKage`Blog](https://blog.iskage.online/) 更新相关项目和学习资料。