import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime


# 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)


# 训练GAN
def train(generator, discriminator, dataloader, criterion, optimizer_G, optimizer_D, epochs, latent_dim, device):
    gen_losses = []
    dis_losses = []
    g_total_loss = 0.0
    d_total_loss = 0.0

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # 准备数据
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # 训练判别器
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = generator(z).detach()

            real_loss = criterion(discriminator(real_imgs), real_labels)
            fake_loss = criterion(discriminator(fake_imgs), fake_labels)
            d_loss = real_loss + fake_loss

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            d_total_loss += d_loss.item()

            # 训练生成器
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            generated_imgs = generator(z)
            g_loss = criterion(discriminator(generated_imgs), real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            g_total_loss += g_loss.item()

        d_avg_loss = d_total_loss / len(dataloader)
        g_avg_loss = g_total_loss / len(dataloader)
        gen_losses.append(g_avg_loss)
        dis_losses.append(d_avg_loss)

        print(f"Epoch [{epoch+1:>2}/{epochs}],   "
              f"Loss D: {d_avg_loss:.4f}, G: {g_avg_loss:.4f}")

        # 每个epoch保存生成的图像
        save_generated_images(epoch, generator, latent_dim, device)
    return gen_losses, dis_losses


def save_generated_images(epoch, generator, latent_dim, device):
    z = torch.randn(64, latent_dim, 1, 1, device=device)
    generated_imgs = generator(z).to(device).detach()
    grid = torchvision.utils.make_grid(generated_imgs, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(f"./data/A3/epoch_{epoch+1}.png")
    plt.close()


# def generate_new_image(generator, latent_dim, device):
#     generator.eval()
#     with torch.no_grad():  # 禁用梯度计算
#         z = torch.randn(1, latent_dim, 1, 1, device=device)  # 随机噪声
#         generated_img = generator(z).to(device)  # 生成图像
#         generated_img = (generated_img + 1) / 2  # 将[-1, 1]范围归一化到[0, 1]
#
#         grid = torchvision.utils.make_grid(generated_img, normalize=True)
#         plt.figure(figsize=(4, 4))
#         plt.imshow(grid.permute(1, 2, 0))  # 转换为HWC格式
#         plt.axis("off")
#         # plt.title("Generated Image")
#         plt.savefig("./data/A3/new_generated_image.png")
#         plt.close()


def generate_new_images(generator, device, latent_dim, num_images):
    """
    生成并保存新的图像。

    Args:
        generator: 训练好的生成器模型。
        latent_dim: 隐空间的维度大小。
        num_images: 需要生成的图像数量。
    """
    generator.eval()  # 切换到评估模式

    # 随机生成噪声输入
    z = torch.randn(num_images, latent_dim, 1, 1, device=device)

    # 生成图像
    with torch.no_grad():
        generated_imgs = generator(z).to(device)

    # 将生成的图像组织成网格
    grid = torchvision.utils.make_grid(generated_imgs, normalize=True, nrow=int(num_images ** 0.5))

    # 保存图像
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("./data/A3/new_generated_images.png")
    plt.close()
    print(f"generating successfully!")


if __name__ == "__main__":
    # 参数设置
    latent_dim = 100
    channels = 3
    batch_size = 256
    epochs = 20
    lr = 0.001
    betas = (0.5, 0.999)

    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * channels, [0.5] * channels)
    ])

    train_loader = DataLoader(
        datasets.CIFAR10("./CIFAR10", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator(latent_dim, channels).to(device)
    dis = Discriminator(channels).to(device)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_D = optim.Adam(dis.parameters(), lr=lr, betas=betas)

    gen_loss, dis_loss = train(gen, dis, train_loader, criterion, opt_G, opt_D, epochs, latent_dim, device)

    current_time = datetime.now().strftime("%H%M")
    pd.DataFrame(gen_loss).to_csv(f'data/A3/{current_time}_gen_loss.csv', index=False, header=False)
    pd.DataFrame(dis_loss).to_csv(f'data/A3/{current_time}_dis_loss.csv', index=False, header=False)

    torch.save(gen.state_dict(), f"./model/A3/{current_time}_Generator.pth")
    torch.save(dis.state_dict(), f"./model/A3/{current_time}_Discriminator.pth")
