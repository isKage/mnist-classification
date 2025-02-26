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
