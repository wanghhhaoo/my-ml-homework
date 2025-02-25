import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pandas as pd
import time
from datetime import datetime


class MLP(nn.Module):
    def __init__(self, inp, oup):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inp, 512)  # 输入层
        self.fc2 = nn.Linear(512, 256)      # 隐藏层
        self.fc3 = nn.Linear(256, oup)       # 输出层
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)           # 展平图像
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


class LoR(nn.Module):
    def __init__(self, inp, oup):
        super(LoR, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(inp, oup)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x


# 4. 训练模型
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []  # 记录每轮的训练损失
    train_accuracies = []  # 记录每轮的训练准确率
    val_losses = []  # 记录每轮的验证损失
    val_accuracies = []  # 记录每轮的验证准确率

    for epoch in range(num_epochs):
        start_time = time.time()

        # 模型训练
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # 正则化强度
        l1_lambda = 0.01
        l2_lambda = 0.01

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # 前向传播
            y_pre = model(x)
            loss = criterion(y_pre, y)

            # L1 正则化项
            l1_norm = sum(torch.sum(torch.abs(param)) for param in model.parameters())

            # L2 正则化项
            l2_norm = sum(torch.sum(param ** 2) for param in model.parameters())

            loss = loss + l1_lambda * l1_norm

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # 累计损失
            _, predicted = torch.max(y_pre, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        end_time = time.time()

        # 计算训练集损失和准确率
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch: [{epoch + 1:>2}/{num_epochs}]")
        print(f"Training   -- Loss: {total_loss / len(train_loader):.4f},  Accuracy: {accuracy:.4f},  "
              f"Time: {end_time - start_time:.3f}")

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

    return train_losses, train_accuracies, val_losses, val_accuracies


# 5. 评估模型性能
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pre = model(x)
            _, predicted = torch.max(y_pre, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    print(f"Test -- Accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    batch_size = 512
    lr = 0.001
    epoch = 50

    # 1. 导入数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_set = datasets.CIFAR10("./CIFAR10", train=True, transform=transform, download=True)
    test_set = datasets.CIFAR10("./CIFAR10", train=False, transform=transform, download=True)

    # 随机种子
    torch.manual_seed(42)

    # 设定训练集和验证集的比例，通常训练集占 80%，验证集占 20%
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    # 使用 random_split 划分训练集和验证集
    train_subset, val_subset = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    input_size = 3 * 32 * 32  # CIFAR-10 图片尺寸是 3x32x32
    num_classes = 10

    mod = 'mlp'
    if mod == 'mlp':
        model = MLP(input_size, num_classes)
    elif mod == 'lor':
        model = LoR(input_size, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. 设置损失函数和优化器 --------------------------------------------------------------------
    cri = 'ce'  # 'mse'
    if cri == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif cri == 'mse':
        criterion = nn.MSELoss()

    opt = 'adam'  # 可以选择 'sgd', 'sgdm', 'adam', 'adagrad'
    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt == 'momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    # ----------------------------------------------------------------------------------------

    train_loss, train_acc, val_loss, val_acc = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=epoch)

    # 记录当前的电脑时间，只显示小时和分钟20
    current_time = datetime.now().strftime("%H%M")

    pd.DataFrame(train_loss).to_csv(f'data/A1/{current_time}_train_loss_{cri}_{opt}.csv', index=False, header=False)
    pd.DataFrame(train_acc).to_csv(f'data/A1/{current_time}_train_acc_{cri}_{opt}.csv', index=False, header=False)
    pd.DataFrame(val_loss).to_csv(f'data/A1/{current_time}_val_loss_{cri}_{opt}.csv', index=False, header=False)
    pd.DataFrame(val_acc).to_csv(f'data/A1/{current_time}_val_acc_{cri}_{opt}.csv', index=False, header=False)

    print('---------- Training End ----------')
    # 评估模型
    test(model, test_loader)

    # 6. 保存模型
    torch.save(model.state_dict(), f"./model/A1/{current_time}_MLP.pth")
