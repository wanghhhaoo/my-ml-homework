import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
from datetime import datetime


class LeNet(nn.Module):
    def __init__(self, normalization):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # 输入通道为 3（RGB）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 是卷积后的尺寸
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # CIFAR-10 有 10 个类别
        self.dropout = nn.Dropout(p=0.5)

        # 根据选择的归一化方式初始化归一化层
        if normalization == 'batch':
            self.norm1 = nn.BatchNorm2d(6)
            self.norm2 = nn.BatchNorm2d(16)
        elif normalization == 'layer':
            self.norm1 = nn.LayerNorm([6, 28, 28])
            self.norm2 = nn.LayerNorm([16, 10, 10])
        elif normalization == 'group':
            self.norm1 = nn.GroupNorm(2, 6)  # 2组
            self.norm2 = nn.GroupNorm(4, 16)  # 4组
        elif normalization == 'instance':
            self.norm1 = nn.InstanceNorm2d(6)
            self.norm2 = nn.InstanceNorm2d(16)
        else:
            raise ValueError("Unsupported normalization type")

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)  # 展平为一维
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class LeNet2(nn.Module):
    def __init__(self):
        super(LeNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # 输入通道为 3（RGB）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 是卷积后的尺寸
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # CIFAR-10 有 10 个类别
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)  # 展平为一维
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class LeNet12(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet12, self).__init__()
        # 输入：3 通道 RGB 图像
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # CIFAR-10 的图像为 32x32，经过两次池化变为 8x8
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 卷积层 + BatchNorm + ReLU + 池化
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # 展平
        x = x.view(x.size(0), -1)  # Flatten
        # 全连接层 + ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 最后一层不需要激活函数
        return x


# 训练模型
def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses = []  # 记录每轮的训练损失
    train_accuracies = []  # 记录每轮的训练准确率
    val_losses = []  # 记录每轮的验证损失
    val_accuracies = []  # 记录每轮的验证准确率

    model.train()  # 设置模型为训练模式
    for epoch in range(epochs):  # 训练轮次
        start_time = time.time()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1:>2}/{epochs}]")
        try:
            for x, y in progress_bar:
                x, y = x.to(device), y.to(device)  # 将数据加载到设备

                z = model(x)  # 前向传播
                loss = criterion(z, y)  # 计算损失
                # loss += l1_regularization(model, lambda_l1=1e-5)  # L1
                # loss += l1_l2_regularization(model, lambda_l1=1e-5, lambda_l2=1e-4)  # L1 + L2

                optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播计算梯度
                optimizer.step()  # 更新模型参数

                total_loss += loss.item()  # 累积损失
                _, predicted = torch.max(z, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        finally:
            progress_bar.close()  # 确保进度条正确关闭

        # for x, y in train_loader:
        #     x, y = x.to(device), y.to(device)  # 将数据加载到设备
        #
        #     z = model(x)  # 前向传播
        #     loss = criterion(z, y)  # 计算损失
        #
        #     optimizer.zero_grad()  # 清空梯度
        #     loss.backward()  # 反向传播计算梯度
        #     optimizer.step()  # 更新模型参数
        #
        #     total_loss += loss.item()  # 累积损失
        #     _, predicted = torch.max(z, 1)
        #     total += y.size(0)
        #     correct += (predicted == y).sum().item()

        end_time = time.time()

        # 计算训练集损失和准确率
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Training   -- Loss: {total_loss / len(train_loader):.4f},  Accuracy: {accuracy:.4f},  "
              f"Time: {end_time - start_time:.3f}")

        val_loss, val_accuracy = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    return train_accuracies, train_losses, val_accuracies, val_losses


def validate(model, val_loader, criterion):
    val_losses = []  # 记录每轮的验证损失
    val_accuracies = []  # 记录每轮的验证准确率

    # 验证模型
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            y_pre = model(x)
            loss = criterion(y_pre, y)
            val_loss += loss.item()
            _, predicted = torch.max(y_pre, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    # 计算验证集损失和准确率
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Validation -- Loss: {val_loss:.4f},  Accuracy: {val_accuracy:.4f}')

    return val_accuracies, val_losses


# 测试模型
def test(model):
    model.eval()  # 设置模型为评估模式
    correct = 0  # 正确样本数
    total = 0  # 总样本数
    with torch.no_grad():  # 禁用梯度计算
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)  # 将数据加载到设备
            z = model(x)  # 前向传播
            _, y_pre = torch.max(z, 1)  # 获取预测的类别
            total += y.size(0)  # 累积样本数
            correct += (y_pre == y).sum().item()  # 累积正确分类数

    accuracy = 100 * correct / total
    print(f"Test -- Accuracy: {accuracy:.2f}")


def ensemble_predict(models, data_loader):
    """模型集成预测"""
    outputs = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                preds.append(torch.softmax(model(inputs), dim=1))
        outputs.append(torch.cat(preds))
    # 对多个模型的预测结果取平均
    final_output = torch.mean(torch.stack(outputs), dim=0)
    return final_output


# L1 正则化
def l1_regularization(model, lambda_l1=1e-5):
    l1_norm = sum(torch.sum(torch.abs(param)) for param in model.parameters() if param.requires_grad)
    return lambda_l1 * l1_norm


# L1 和 L2
def l1_l2_regularization(model, lambda_l1=1e-5, lambda_l2=1e-4):
    l1_norm = sum(torch.sum(torch.abs(param)) for param in model.parameters() if param.requires_grad)
    l2_norm = sum(torch.sum(param ** 2) for param in model.parameters() if param.requires_grad)
    return lambda_l1 * l1_norm + lambda_l2 * l2_norm


if __name__ == '__main__':
    hps = {'lr': 0.001, 'batch_size': 128, 'epochs': 10}
    lr = hps['lr']
    batch_size = hps['batch_size']
    epochs = hps['epochs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fg_1 = ['norm', 'layer', 'reg', 'data']
    fg_1 = 'layer_12'
    normalizations = ['batch', 'layer', 'group', 'instance']

    # 数据预处理
    transform = transforms.Compose([
        # 归一化
        transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
        transforms.Normalize([0.5], [0.5])  # 归一化像素值到 [-1, 1]
    ])
    # transform = transforms.Compose([
    #     # 数据增强
    #     transforms.RandomHorizontalFlip(),  # 随机水平翻转
    #     transforms.RandomCrop(32, padding=4),  # 随机裁剪图像并填充
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    # 加载数据集
    train_set = datasets.CIFAR10("./CIFAR10", train=True, transform=transform, download=True)
    test_set = datasets.CIFAR10("./CIFAR10", train=False, transform=transform, download=True)

    torch.manual_seed(42)

    # 设定训练集和验证集的比例，通常训练集占 80%，验证集占 20%
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    # 使用 random_split 划分训练集和验证集
    train_subset, val_subset = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # model = LeNet2()
    model = LeNet12()
    cri = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    t_acc, t_loss, v_acc, v_loss = train(model, train_loader, val_loader, cri, opt, epochs)

    # 模型集成
    # model_1 = LeNet2()
    # model_2 = LeNet2()
    # cri = nn.CrossEntropyLoss()
    # opt_1 = optim.Adam(model_1.parameters(), lr=lr)
    # opt_2 = optim.Adam(model_2.parameters(), lr=lr)
    # t_acc_1, t_loss_1, v_acc_1, v_loss_1 = train(model_1, train_loader, val_loader, cri, opt_1, epochs)
    # t_acc_2, t_loss_2, v_acc_2, v_los_2s = train(model_2, train_loader, val_loader, cri, opt_2, epochs)
    # final_predictions = ensemble_predict([model_1, model_2], val_loader)

    current_time = datetime.now().strftime("%H%M")
    pd.DataFrame(t_acc).to_csv(f"data/A2/acc_t_{fg_1}.csv", index=False, header=False)
    pd.DataFrame(t_loss).to_csv(f"data/A2/loss_t_{fg_1}.csv", index=False, header=False)
    pd.DataFrame(v_acc).to_csv(f"data/A2/acc_v_{fg_1}.csv", index=False, header=False)
    pd.DataFrame(v_loss).to_csv(f"data/A2/loss_v_{fg_1}.csv", index=False, header=False)

    torch.save(model.state_dict(), f'./model/A2/{current_time}_LeNet_{fg_1}.pth')

    # for norm in normalizations:
    #     print(f"Testing normalization: {norm}")
    #     model = LeNet(normalization=norm)
    #     cri = nn.CrossEntropyLoss()
    #     opt = optim.Adam(model.parameters(), lr=lr)
    #     # 训练和验证模型，记录结果
    #     t_acc, t_loss, v_acc, v_loss = train(model, train_loader, val_loader, cri, opt, epochs)
    #
    #     current_time = datetime.now().strftime("%H%M")
    #     pd.DataFrame(v_acc).to_csv(f"data/A2/acc_{current_time}_{fg_1[0]}_{norm}.csv", index=False, header=False)
    #     pd.DataFrame(v_loss).to_csv(f"data/A2/loss_{current_time}_{fg_1[0]}_{norm}.csv", index=False, header=False)
    #
    #     torch.save(model.state_dict(), f'./model/A2/LeNet_{fg_1[0]}_{norm}.pth')

    print('---------- Training End ----------')
