'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-05-03 14:27:01
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-05-04 03:12:33
FilePath: \Leaves-Segmentation-Challenge\src\self_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn

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

class Transformer_VAE(nn.Module):
    def __init__(self):
        super(Transformer_VAE, self).__init__()
        self.groupnorm = nn.GroupNorm(1, 3)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=500,
                nhead=10,
                dropout=0.1
            ),
            num_layers=6
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=500,
                nhead=10,
                dropout=0.1
            ),
            num_layers=6
        )
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        x = self.groupnorm(x)
        x = x[:, 0, :, :]
        # print(x.shape)
        x = self.encoder(x)
        tgt = torch.zeros_like(x)
        x = self.decoder(tgt, x)
        x = x.unsqueeze(0).permute(1, 0, 2, 3)
        return x