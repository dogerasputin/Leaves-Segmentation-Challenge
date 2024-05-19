'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-05-03 14:27:01
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-05-19 16:02:27
FilePath: \Leaves-Segmentation-Challenge\src\self_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models

class MLP_VAE(nn.Module):
    def __init__(self, input_dim=530*500*3, latent_dim=1024, hidden_dim=1024, last_dim=530*500):
        super(MLP_VAE, self).__init__()
        self.flatten = nn.Flatten()
        self.norm = nn.BatchNorm1d(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Linear(hidden_dim, last_dim),
            nn.Hardsigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.norm(x)
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        # reshape
        x = x.view(-1, 1, 530, 500)
        return x, mu, logvar
    
class EVANet(nn.Module):
    def __init__(self):
        super(EVANet, self).__init__()
        self.model = timm.create_model('eva02_small_patch14_224')
        self.adaptive_pool = nn.AdaptiveAvgPool2d((224, 224))
        
    def forward(self, x):
        x = self.model.forward_features(x)
        # downsample
        # x = F.adaptive_avg_pool2d(x, (224, 224))
        x = self.adaptive_pool(x)
        # (batch, 224, 224) -> (batch, 1, 224, 224)
        x = x.unsqueeze(1)
        return x

class HieraTinyNet(nn.Module):
    """
    HieraTinyNet(Classification) model
    """
    def __init__(self):
        super(HieraTinyNet, self).__init__()
        self.model = timm.create_model('hiera_tiny_224')
        self.out_conv = nn.Conv2d(1, 1, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((224, 224))
        
    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = self.out_conv(x)
        # print(x.shape)
        x = self.adaptive_pool(x)
        return x

class ConVIT(nn.Module):
    def __init__(self):
        super(ConVIT, self).__init__()
        self.model = timm.create_model('convit_small')
        self.out_conv = nn.Conv2d(1, 1, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((224, 224))
        
    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.unsqueeze(1)
        x = self.out_conv(x)
        x = self.adaptive_pool(x)
        return x

class Deeplab(nn.Module):
    def __init__(self):
        super(Deeplab, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=1)
        self.model = models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=20)

    def forward(self, x):
        x = self.conv1(x)
        x = self.model(x)
        x = x['out']
        return x

class DeeplabGRU(nn.Module):
    def __init__(self, num_classes=20, hidden_dim=256, num_layers=1):
        super(DeeplabGRU, self).__init__()
        # 初始化 DeeplabV3 模型
        self.deeplab = models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=num_classes, pretrained=False)
        # 2D 卷積層，用於特徵調整
        self.conv1 = nn.Conv2d(1, 3, kernel_size=1)
        # GRU 層，用於處理時間序列數據
        self.gru = nn.GRU(input_size=num_classes, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        # 分類層
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x 應該是一個形狀為 (batch_size, time_steps, channels, height, width) 的五維張量
        batch_size, time_steps, C, H, W = x.size()
        # 調整 x 的形狀以適應 deeplab 模型
        x = x.view(batch_size * time_steps, C, H, W)
        x = self.conv1(x)
        x = self.deeplab(x)["out"]
        # 調整 x 的形狀以適應 GRU 的輸入
        x = x.view(batch_size, time_steps, -1)
        x, _ = self.gru(x)
        # 取 GRU 輸出的最後一個時間步
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class TransformerNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(TransformerNET, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.groupnorm = nn.GroupNorm(1, 3)
        self.model = nn.Transformer(
            d_model=224,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.out_conv = nn.Conv2d(1, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.groupnorm(x)
        x = x[:, 0, :, :]
        x = self.model(x, x)
        x = x.unsqueeze(1)
        x = self.out_conv(x)
        return x


class RecurrentUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(RecurrentUnet, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.model1 = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',in_channels=3, out_channels=1, init_features=4, pretrained=False)
        self.model2 = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',in_channels=3, out_channels=1, init_features=4, pretrained=False)
        self.model3 = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',in_channels=3, out_channels=1, init_features=4, pretrained=False)
        self.model4 = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',in_channels=3, out_channels=1, init_features=4, pretrained=False)
        self.model5 = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',in_channels=3, out_channels=1, init_features=4, pretrained=False)
        self.out_conv = nn.Conv2d(1, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x4 = self.model4(x)
        x5 = self.model5(x)
        x = x1 + x2 + x3 + x4 + x5
        x = self.out_conv(x)
        return x

if __name__ == "__main__": # Test
    model = RecurrentUnet(in_channels=1,out_channels=20)
    print(model)
    x = torch.randn(32, 1, 224, 224)
    y = model(x) # outside
    print(y.shape)
    print("Test passed!")