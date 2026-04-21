from __future__ import annotations

import torch
from torch import nn

try:
    from torchvision.models import efficientnet_b0, efficientnet_b2
    _TORCHVISION_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency path
    efficientnet_b0 = None
    efficientnet_b2 = None
    _TORCHVISION_IMPORT_ERROR = exc


SUPPORTED_MODEL_NAMES = ("custom_cnn", "efficientnet_b0", "efficientnet_b2")


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


class _CustomBirdCLEFNet(nn.Module):
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


def _replace_first_conv_to_single_channel(conv: nn.Conv2d) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        1,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        if conv.weight.shape[1] == 3:
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        elif conv.weight.shape[1] == 1:
            new_conv.weight.copy_(conv.weight)
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def _build_efficientnet(model_name: str, num_classes: int, dropout: float) -> nn.Module:
    if efficientnet_b0 is None or efficientnet_b2 is None:
        details = ""
        if _TORCHVISION_IMPORT_ERROR is not None:
            details = f" Original import error: {_TORCHVISION_IMPORT_ERROR!r}"
        raise ImportError(
            "torchvision is required for EfficientNet backbones. "
            "Please install torchvision, or use --model-name custom_cnn."
            + details
        )

    model_name = str(model_name).lower()
    if model_name == "efficientnet_b0":
        backbone = efficientnet_b0(weights=None)
    elif model_name == "efficientnet_b2":
        backbone = efficientnet_b2(weights=None)
    else:
        raise ValueError(
            f"Unsupported EfficientNet model name: {model_name}. "
            f"Supported: {SUPPORTED_MODEL_NAMES}"
        )

    first_conv = backbone.features[0][0]
    if isinstance(first_conv, nn.Conv2d):
        backbone.features[0][0] = _replace_first_conv_to_single_channel(first_conv)

    if isinstance(backbone.classifier, nn.Sequential) and len(backbone.classifier) >= 2:
        if isinstance(backbone.classifier[0], nn.Dropout):
            backbone.classifier[0] = nn.Dropout(p=dropout, inplace=True)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)
    else:  # pragma: no cover - defensive path
        raise RuntimeError("Unexpected EfficientNet classifier structure.")

    return backbone


class BirdCLEFNet(nn.Module):
    def __init__(
        self, num_classes: int, dropout: float = 0.3, model_name: str = "custom_cnn"
    ) -> None:
        super().__init__()
        model_name = str(model_name).lower()
        self.model_name = model_name

        if model_name == "custom_cnn":
            self.backbone = _CustomBirdCLEFNet(num_classes=num_classes, dropout=dropout)
        elif model_name in {"efficientnet_b0", "efficientnet_b2"}:
            self.backbone = _build_efficientnet(
                model_name=model_name, num_classes=num_classes, dropout=dropout
            )
        else:
            raise ValueError(
                f"Unsupported model_name={model_name}. "
                f"Supported: {SUPPORTED_MODEL_NAMES}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
