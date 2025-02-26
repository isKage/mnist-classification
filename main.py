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
