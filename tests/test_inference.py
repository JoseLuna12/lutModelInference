from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lutinferencemodel import Predictor, load_image_float, save_preview_image
from lutcore import apply_image

FIXTURE_DIR = Path(__file__).parent / "data"
SOURCE_IMAGE = FIXTURE_DIR / "_UwxuO3BVuM.jpg"
TARGET_IMAGE = FIXTURE_DIR / "0095.png"


@pytest.fixture(scope="session")
def predictor_cpu() -> Predictor:
    # Force CPU to make the test deterministic and avoid GPU availability differences.
    return Predictor(device="cpu", pretrained_backbone=False)


@pytest.fixture(scope="session")
def prediction(predictor_cpu: Predictor):
    return predictor_cpu.predict(SOURCE_IMAGE, apply_lut=True)


def test_predictor_exports_lut(tmp_path, predictor_cpu: Predictor, prediction):
    assert SOURCE_IMAGE.exists()
    lut_path, weights_path = predictor_cpu.save_outputs(SOURCE_IMAGE, prediction, output_dir=tmp_path)
    assert lut_path.exists(), "LUT file should be written"
    assert weights_path.exists(), "Weights JSON should be written"
    assert prediction.weights.size > 0
    assert prediction.applied_image is not None
    assert prediction.applied_image.shape[2] == 3


def test_apply_predicted_lut_to_target(tmp_path, prediction):
    assert TARGET_IMAGE.exists()
    target = load_image_float(TARGET_IMAGE)
    applied = apply_image(target, prediction.lut)
    assert applied.shape == target.shape

    # Persist a quick preview to ensure writing works end-to-end.
    preview_path = tmp_path / "target_lut_preview.png"
    save_preview_image(applied, preview_path)
    assert preview_path.exists()


def test_save_outputs_respects_flags(tmp_path, predictor_cpu: Predictor, prediction):
    # Only weights
    lut_path, weights_path = predictor_cpu.save_outputs(
        SOURCE_IMAGE, prediction, output_dir=tmp_path, save_lut=False, save_weights=True
    )
    assert lut_path is None
    assert weights_path and weights_path.exists()

    # Only LUT
    lut_path2, weights_path2 = predictor_cpu.save_outputs(
        SOURCE_IMAGE, prediction, output_dir=tmp_path, save_lut=True, save_weights=False
    )
    assert weights_path2 is None
    assert lut_path2 and lut_path2.exists()

    with pytest.raises(ValueError):
        predictor_cpu.save_outputs(SOURCE_IMAGE, prediction, output_dir=tmp_path, save_lut=False, save_weights=False)


def test_save_preview_image_roundtrip(tmp_path):
    # Use the real fixture image to ensure the helper works on actual data.
    arr = load_image_float(SOURCE_IMAGE)
    expected = (arr * 255.0).clip(0, 255).astype(np.uint8)

    preview_path = tmp_path / "preview.png"
    saved_path = save_preview_image(arr, preview_path)
    assert saved_path == preview_path
    assert preview_path.exists()

    loaded = np.asarray(Image.open(preview_path))
    np.testing.assert_array_equal(loaded, expected)
