# lutinferencemodel

Standalone inference-only package to turn an image into a LUT using a trained EfficientNet-B0 -> PCA-weights model.

## Layout
- `src/lutinferencemodel/vision.py` — EfficientNet head + transforms.
- `src/lutinferencemodel/predictor.py` — `Predictor` API, artifact loading, weights->LUT conversion.
- `src/lutinferencemodel/artifacts/` — bundled checkpoint + PCA artifacts.
- `src/lutinferencemodel/cli.py` — CLI entrypoint (`lutinferencemodel` script).

## Quickstart
1) Install deps (needs network access for torch, torchvision, lutcore Git dependency):
```bash
uv sync
```
2) Run a prediction from the CLI (see flags below):
```bash
uv run lutinferencemodel path/to/image.png --out outputs --preview
```

Outputs:
- `outputs/<stem>_pred.cube` (reconstructed LUT)
- `outputs/<stem>_weights.json` (denormalized PCA weights)
- Optional `outputs/<stem>_preview.png` if `--preview` is set.

## CLI usage
```
uv run lutinferencemodel <image> [--out outputs]
                             [--artifacts src/lutinferencemodel/artifacts]
                             [--checkpoint best.pt]
                             [--pca-components pca_components.npy]
                             [--pca-mean pca_mean.npy]
                             [--weight-norm weight_normalization.json]
                             [--vision-config vision.json]
                             [--device cpu|cuda|mps]
                             [--no-apply]
                             [--preview]
```
- `--no-apply` skips applying the LUT; prediction still returns weights.
- `--preview` writes a preview PNG (needs apply enabled).
- Artifacts default to the packaged set under `src/lutinferencemodel/artifacts/`.

## Python API usage
```python
from lutinferencemodel import Predictor, load_image_float

pred = Predictor()  # uses packaged artifacts by default
result = pred.predict("path/to/image.png", apply_lut=True)
lut_path, weights_path = pred.save_outputs("path/to/image.png", result, output_dir="outputs")

# Apply the predicted LUT to another image (numpy/PIL/path all supported).
target = load_image_float("path/to/target.png")
applied = result.lut.apply(target)  # or lutcore.apply_image(target, result.lut)
```

## Tests
Sample regression tests (uses bundled sample images):
```bash
uv run pytest
```
