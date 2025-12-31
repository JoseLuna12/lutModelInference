from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .predictor import Predictor, PredictionResult, select_device


def _device_from_arg(arg: Optional[str]):
    if arg is None:
        return None
    name = arg.lower()
    if name in ("cpu", "cuda", "mps"):
        return torch.device(name)
    raise ValueError(f"Unsupported device: {arg}")


def _save_preview(applied: np.ndarray, path: Path) -> None:
    arr = np.clip(applied * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def run(image: Path, args) -> None:
    device = args.device or select_device()
    predictor = Predictor(
        artifact_dir=args.artifacts,
        checkpoint_path=args.checkpoint,
        pca_components_path=args.pca_components,
        pca_mean_path=args.pca_mean,
        weight_norm_path=args.weight_norm,
        vision_config_path=args.vision_config,
        device=device,
        pretrained_backbone=False,
    )
    result: PredictionResult = predictor.predict(image, apply_lut=not args.no_apply)
    lut_path = weights_path = None
    if args.save_lut or args.save_weights:
        lut_path, weights_path = predictor.save_outputs(
            image,
            result,
            output_dir=args.out,
            save_lut=args.save_lut,
            save_weights=args.save_weights,
        )

    if args.save_lut and lut_path is not None:
        print(f"Saved LUT      -> {lut_path}")
    if args.save_weights and weights_path is not None:
        print(f"Saved weights  -> {weights_path}")
    if args.print_weights:
        print(json.dumps({"weights": result.weights.tolist()}))

    print(f"Weights shape  -> {result.weights.shape}")
    if result.applied_image is not None and args.preview:
        preview_path = Path(args.out) / f"{image.stem}_preview.png"
        _save_preview(result.applied_image, preview_path)
        print(f"Preview image  -> {preview_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict LUTs for images using a trained EfficientNet PCA model.")
    p.add_argument("image", type=Path, help="Path to input image")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="Directory to write LUT/weights (default: outputs)")
    p.add_argument("--artifacts", type=Path, default=None, help="Directory containing checkpoint + PCA artifacts (defaults to packaged ones)")
    p.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path")
    p.add_argument("--pca-components", type=Path, default=None, help="Override PCA components path")
    p.add_argument("--pca-mean", type=Path, default=None, help="Override PCA mean path")
    p.add_argument("--weight-norm", type=Path, default=None, help="Override weight normalization path")
    p.add_argument("--vision-config", type=Path, default=None, help="Optional vision config JSON (defaults to EfficientNet-B0 settings)")
    p.add_argument("--device", type=_device_from_arg, default=None, help="Force device: cpu|cuda|mps (default: auto)")
    p.add_argument("--no-lut", dest="save_lut", action="store_false", help="Skip writing LUT .cube output")
    p.add_argument("--no-weights", dest="save_weights", action="store_false", help="Skip writing weights JSON")
    p.add_argument("--print-weights", action="store_true", help="Print predicted weights (JSON) to stdout")
    p.add_argument("--no-apply", action="store_true", help="Skip applying LUT to produce preview array")
    p.add_argument("--preview", action="store_true", help="Save preview PNG with LUT applied (requires --no-apply off)")
    p.set_defaults(save_lut=True, save_weights=True)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args.image, args)


if __name__ == "__main__":
    main()
