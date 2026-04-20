from __future__ import annotations

import torch
from torch import nn


class ConvBnAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualUnit(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBnAct(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.SiLU(inplace=True)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class BirdCLEFNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = ConvBnAct(1, 32, stride=1)
        self.stage1 = nn.Sequential(
            ResidualUnit(32, 64, stride=2),
            ResidualUnit(64, 64, stride=1),
        )
        self.stage2 = nn.Sequential(
            ResidualUnit(64, 128, stride=2),
            ResidualUnit(128, 128, stride=1),
        )
        self.stage3 = nn.Sequential(
            ResidualUnit(128, 256, stride=2),
            ResidualUnit(256, 256, stride=1),
        )
        self.stage4 = nn.Sequential(
            ResidualUnit(256, 384, stride=2),
            ResidualUnit(384, 384, stride=1),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(384 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        avg_pool = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        max_pool = torch.nn.functional.adaptive_max_pool2d(x, output_size=1).flatten(1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.dropout(x)
        return self.head(x)
