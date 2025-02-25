import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import time
import pandas as pd
from datetime import datetime


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # 输入通道为 3（RGB）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 是卷积后的尺寸
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # CIFAR-10 有 10 个类别
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # 展平为一维
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 训练模型
def train(model, train_loader, criterion, optimizer, epochs):
    t_accs = []
    t_losses = []

    model.train()  # 设置模型为训练模式
    for epoch in range(epochs):  # 训练轮次
        start_time = time.time()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)  # 将数据加载到设备

            z = model(x)  # 前向传播
            loss = criterion(z, y)  # 计算损失

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            total_loss += loss.item()  # 累积损失
            _, predicted = torch.max(z, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        end_time = time.time()

        # 计算训练集损失和准确率
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch: [{epoch + 1:>2}/{epochs}] -- "
              f"Loss: {avg_loss:.4f},  "
              f"Accuracy: {accuracy:.4f},  "
              f"Time: {end_time - start_time:.3f}")
        t_accs.append(accuracy)
        t_losses.append(avg_loss)

    return t_accs, t_losses


def validate(model, val_loader, criterion):
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
    print(f'Validation -- Loss: {val_loss:.4f},  Accuracy: {val_accuracy:.4f}')

    return val_accuracy, val_loss


# 测试模型
def test(model, test_loader):
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


# 交叉验证函数
def cross_validate(dataset, hyperparams, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    best_params = None
    best_score = float('inf')

    for lr in hyperparams['lr']:
        for batch_size in hyperparams['batch_size']:
            fold_scores = []
            acc = []
            loss = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
                print(f"Fold {fold + 1},  params: lr: {lr}, batch_size: {batch_size}")

                # 数据子集
                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, val_idx)

                # 数据加载器
                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

                torch.manual_seed(42)
                # 初始化模型、损失函数和优化器
                model = LeNet()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # 训练与验证
                t_acc, t_loss = train(model, train_loader, criterion, optimizer, hyperparams['epochs'])
                acc.append(t_acc)
                loss.append(t_loss)

                v_acc, v_loss = validate(model, val_loader, criterion)
                fold_scores.append(v_loss)

            acc = np.sum(np.array(acc), axis=0)
            loss = np.sum(np.array(loss), axis=0)

            pd.DataFrame(acc).to_csv(f"data/A2/acc_{lr}_{batch_size}.csv", index=False, header=False)
            pd.DataFrame(loss).to_csv(f"data/A2/loss_{lr}_{batch_size}.csv", index=False, header=False)

            # 平均性能
            avg_score = np.mean(fold_scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = {'lr': lr, 'batch_size': batch_size}

    return best_params, best_score


if __name__ == '__main__':
    hps = {
        'lr': [0.001, 0.005, 0.01],
        # 'hidden_size': [32, 64, 128],
        'batch_size': [64, 128, 256],
        'epochs': 10
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),  # 随机水平翻转
        # transforms.RandomCrop(32, padding=4),  # 随机裁剪图像并填充
        transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
        transforms.Normalize([0.5], [0.5])  # 归一化像素值到 [-1, 1]
    ])

    dataset = datasets.CIFAR10("./CIFAR10", train=True, download=True, transform=transform)

    params, score = cross_validate(dataset, hps, 5)
    print(f"The best hyperparams: {params}")
    print(f"The best scores: {score:.4f}")

    print('---------- Training End ----------')
