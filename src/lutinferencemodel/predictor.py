from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from lutcore import LUT, apply_image, unflatten

from .vision import build_efficientnet_pca, build_image_transform, load_vision_config

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
DEFAULT_CHECKPOINT = DEFAULT_ARTIFACT_DIR / "best.pt"
DEFAULT_PCA_COMPONENTS = DEFAULT_ARTIFACT_DIR / "pca_components.npy"
DEFAULT_PCA_MEAN = DEFAULT_ARTIFACT_DIR / "pca_mean.npy"
DEFAULT_WEIGHT_NORM = DEFAULT_ARTIFACT_DIR / "weight_normalization.json"
DEFAULT_LUT_SIZE = 33

ImageLike = Union[str, Path, Image.Image, np.ndarray]


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_artifacts(
    *,
    checkpoint: Path,
    pca_components: Path,
    pca_mean: Path,
    weight_norm: Path,
) -> None:
    missing = [str(p) for p in (checkpoint, pca_components, pca_mean, weight_norm) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")


def load_artifacts(
    *,
    checkpoint: Path,
    pca_components: Path,
    pca_mean: Path,
    weight_norm: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    ensure_artifacts(
        checkpoint=checkpoint,
        pca_components=pca_components,
        pca_mean=pca_mean,
        weight_norm=weight_norm,
    )
    components = torch.from_numpy(np.load(pca_components)).float().to(device)
    mean = torch.from_numpy(np.load(pca_mean)).float().to(device)
    with weight_norm.open("r") as f:
        norm = json.load(f)
    return components, mean, norm, {"checkpoint": checkpoint}


def denorm_weights(weights: torch.Tensor, norm: dict) -> torch.Tensor:
    """Denormalize PCA weights using stored mean/std."""
    mean = torch.tensor(norm["mean"], device=weights.device, dtype=weights.dtype)
    std = torch.tensor(norm["std"], device=weights.device, dtype=weights.dtype)
    return weights * std + mean


def weights_to_lut(
    weights: torch.Tensor,
    components: torch.Tensor,
    mean: torch.Tensor,
    *,
    lut_size: int = DEFAULT_LUT_SIZE,
) -> LUT:
    """Convert a single weight vector (1, N) to a lutcore LUT using PCA artifacts."""
    with torch.no_grad():
        vec = torch.matmul(weights, components) + mean  # (1, D)
        vec_np = vec.squeeze(0).cpu().numpy()
    table = unflatten(vec_np, size=lut_size)
    return LUT(
        table=table,
        size=lut_size,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
    )


def _as_float_hwc(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expected HWC image with 3 channels")
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.ascontiguousarray(arr)


def load_image_float(image: ImageLike) -> np.ndarray:
    """
    Load an image-like (path, PIL Image, or ndarray) into a contiguous float32 HWC array in [0, 1].
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return _as_float_hwc(Image.open(path))
    return _as_float_hwc(image)


@dataclass
class PredictionResult:
    lut: LUT
    applied_image: Optional[np.ndarray]
    weights: np.ndarray


class Predictor:
    """
    Load model + PCA artifacts once, then predict LUTs for images.
    """

    def __init__(
        self,
        *,
        artifact_dir: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
        pca_components_path: Path | str | None = None,
        pca_mean_path: Path | str | None = None,
        weight_norm_path: Path | str | None = None,
        vision_config_path: Path | str | None = None,
        device: torch.device | None = None,
        lut_size: int = DEFAULT_LUT_SIZE,
        pretrained_backbone: bool = False,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device or select_device()
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else DEFAULT_ARTIFACT_DIR
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else self.artifact_dir / "best.pt"
        self.pca_components_path = Path(pca_components_path) if pca_components_path else self.artifact_dir / "pca_components.npy"
        self.pca_mean_path = Path(pca_mean_path) if pca_mean_path else self.artifact_dir / "pca_mean.npy"
        self.weight_norm_path = Path(weight_norm_path) if weight_norm_path else self.artifact_dir / "weight_normalization.json"
        self.lut_size = lut_size

        components, mean, norm, _ = load_artifacts(
            checkpoint=self.checkpoint_path,
            pca_components=self.pca_components_path,
            pca_mean=self.pca_mean_path,
            weight_norm=self.weight_norm_path,
            device=self.device,
        )
        self.components = components
        self.mean = mean
        self.norm = norm

        n_outputs = components.shape[0]
        self.model = build_efficientnet_pca(
            n_outputs=n_outputs,
            pretrained=pretrained_backbone,
            freeze_backbone=False,
        ).to(self.device)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.eval()

        cfg = load_vision_config(vision_config_path)
        self.transform = build_image_transform(cfg, is_train=False)

    def _load_pil(self, image: ImageLike) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray((_as_float_hwc(image) * 255).astype(np.uint8))
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

    def predict(self, image: ImageLike, *, apply_lut: bool = True) -> PredictionResult:
        pil_img = self._load_pil(image)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_norm = self.model(input_tensor)
            pred = denorm_weights(pred_norm, self.norm).to(self.device)

        lut = weights_to_lut(pred, self.components, self.mean, lut_size=self.lut_size)

        applied = None
        if apply_lut:
            base_np = _as_float_hwc(pil_img)
            applied = apply_image(base_np, lut)

        weights_np = pred.squeeze(0).cpu().numpy()
        return PredictionResult(lut=lut, applied_image=applied, weights=weights_np)

    def save_outputs(
        self,
        image_path: Path | str,
        result: PredictionResult,
        *,
        output_dir: Path | str,
    ) -> tuple[Path, Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(image_path).stem
        lut_path = out_dir / f"{base}_pred.cube"
        weights_path = out_dir / f"{base}_weights.json"
        result.lut.export(str(lut_path))
        with weights_path.open("w") as f:
            json.dump({"weights": result.weights.tolist()}, f, indent=2)
        return lut_path, weights_path
