import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
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

            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
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

# 参数设置
latent_dim = 100
img_channels = 3
batch_size = 128
epochs = 50
learning_rate = 0.0002

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * img_channels, [0.5] * img_channels)
])

dataloader = DataLoader(
    datasets.CIFAR10("./CIFAR10", train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(latent_dim, img_channels).to(device)
discriminator = Discriminator(img_channels).to(device)

# 定义优化器和损失函数
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# 训练GAN
def train():
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

            # 训练生成器
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            generated_imgs = generator(z)
            g_loss = criterion(discriminator(generated_imgs), real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            # 打印训练信息
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {i}/{len(dataloader)} \n"
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # 每个epoch保存生成的图像
        save_generated_images(epoch)

def save_generated_images(epoch):
    z = torch.randn(64, latent_dim, 1, 1, device=device)
    generated_imgs = generator(z).cpu().detach()
    grid = torchvision.utils.make_grid(generated_imgs, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(f"generated_images_epoch_{epoch+1}.png")
    plt.close()


if __name__ == "__main__":
    train()
