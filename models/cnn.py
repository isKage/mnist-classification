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
