from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# Default input contract for EfficientNet-B0 (ImageNet-pretrained).
EFFICIENTNET_B0_DEFAULT = {
    "arch": "efficientnet_b0",
    "image_size": [224, 224],
    "normalization": {
        "type": "imagenet",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "augmentation": {
        "random_resized_crop": False,
        "center_crop": True,
        "hflip": False,
    },
}


def load_vision_config(path: str | Path | None) -> dict:
    """Load vision config from JSON; fall back to EfficientNet-B0 defaults when path is None."""
    if path is None:
        return deepcopy(EFFICIENTNET_B0_DEFAULT)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Vision config not found: {path}")
    with p.open("r") as f:
        return json.load(f)


def _resolve_norm(norm_cfg: dict) -> tuple[list[float], list[float]]:
    """Map normalization configs to mean/std lists."""
    norm_type = norm_cfg.get("type", "imagenet")
    if norm_type == "imagenet":
        return norm_cfg.get("mean", EFFICIENTNET_B0_DEFAULT["normalization"]["mean"]), norm_cfg.get(
            "std", EFFICIENTNET_B0_DEFAULT["normalization"]["std"]
        )
    raise ValueError(f"Unsupported normalization type: {norm_type}")


def build_image_transform(cfg: dict, is_train: bool = False):
    """
    Build transforms for vision backbone input.
    - For EfficientNet-B0 defaults: Resize/CenterCrop -> ToTensor -> Normalize(ImageNet).
    - Minimal optional aug: RandomResizedCrop, HorizontalFlip.
    """
    size = tuple(cfg.get("image_size", EFFICIENTNET_B0_DEFAULT["image_size"]))
    norm = cfg.get("normalization", EFFICIENTNET_B0_DEFAULT["normalization"])
    mean, std = _resolve_norm(norm)
    aug = cfg.get("augmentation", EFFICIENTNET_B0_DEFAULT["augmentation"])

    steps: list[transforms.Compose | transforms.Normalize] = []
    if is_train and aug.get("random_resized_crop", False):
        steps.append(transforms.RandomResizedCrop(size[0]))
    else:
        steps.append(transforms.Resize(size))
        if aug.get("center_crop", True):
            steps.append(transforms.CenterCrop(size))

    if is_train and aug.get("hflip", False):
        steps.append(transforms.RandomHorizontalFlip())

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(steps)


class EfficientNetPCA(nn.Module):
    """
    EfficientNet-B0 backbone with a lightweight MLP head to regress PCA weights.
    """

    def __init__(
        self,
        n_outputs: int,
        *,
        pretrained: bool = True,
        head_hidden: Iterable[int] = (256, 128),
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.pool = backbone.avgpool

        in_dim = backbone.classifier[1].in_features
        mlp_layers: list[nn.Module] = []
        dims = (in_dim, *tuple(head_hidden), n_outputs)
        for i in range(len(dims) - 1):
            mlp_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mlp_layers.append(nn.GELU())
                if dropout and dropout > 0:
                    mlp_layers.append(nn.Dropout(dropout))
        self.head = nn.Sequential(*mlp_layers)

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def build_efficientnet_pca(
    n_outputs: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    head_hidden: Iterable[int] = (256, 128),
    dropout: float = 0.1,
) -> EfficientNetPCA:
    """Factory to build the EfficientNet PCA regressor."""
    return EfficientNetPCA(
        n_outputs=n_outputs,
        pretrained=pretrained,
        head_hidden=head_hidden,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
