"""Small proxy models for augmentation screening."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LightUNet(nn.Module):
    def __init__(self, in_chans: int = 1, num_classes: int = 4, base: int = 16):
        super().__init__()
        self.e1 = ConvBlock(in_chans, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.b = ConvBlock(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        x = self.d3(torch.cat([self.u3(b), e3], dim=1))
        x = self.d2(torch.cat([self.u2(x), e2], dim=1))
        x = self.d1(torch.cat([self.u1(x), e1], dim=1))
        return self.head(x)


class FrameCNN(nn.Module):
    def __init__(self, in_chans: int = 1, out_dim: int = 128, base: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, base, 3, stride=2, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CNNGRUEF(nn.Module):
    def __init__(self, in_chans: int = 1, feat_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.frame = FrameCNN(in_chans, feat_dim)
        self.gru = nn.GRU(feat_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, 1))

    def forward(self, video):
        b, t, c, h, w = video.shape
        feat = self.frame(video.reshape(b * t, c, h, w)).reshape(b, t, -1)
        out, _ = self.gru(feat)
        return self.head(out[:, -1]).squeeze(-1)


class SmallMAE(nn.Module):
    def __init__(self, in_chans: int = 1, base: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, base, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, in_chans, 2, stride=2),
            nn.Sigmoid(),
        )

    def random_mask(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
        mask = (torch.rand_like(x[:, :1]) < mask_ratio).float()
        return x * (1.0 - mask), mask

    def forward(self, x, mask_ratio: float = 0.75):
        masked, mask = self.random_mask(x, mask_ratio)
        rec = self.decoder(self.encoder(masked))
        return rec, mask
