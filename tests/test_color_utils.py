import numpy as np

from src.cvpipe.color_utils import (
    rgb_to_lab, lab_to_rgb, normalize_lab, denormalize_lab,
    rgb_to_model_input_target, model_output_to_rgb,
)


def _random_rgb(h=64, w=64, seed=0):
    return np.random.default_rng(seed).random((h, w, 3)).astype(np.float32)


def test_float_roundtrip():
    rgb = _random_rgb()
    back = lab_to_rgb(rgb_to_lab(rgb))
    assert np.abs(rgb - back).max() < 5e-3


def test_uint8_roundtrip():
    rgb8 = (_random_rgb() * 255).round().astype(np.uint8)
    back8 = (lab_to_rgb(rgb_to_lab(rgb8)) * 255).round().astype(np.int16)
    assert np.abs(rgb8.astype(np.int16) - back8).max() <= 1


def test_normalize_roundtrip():
    lab = rgb_to_lab(_random_rgb())
    back = denormalize_lab(normalize_lab(lab))
    assert np.abs(lab - back).max() < 1e-4


def test_normalized_ranges():
    lab_n = normalize_lab(rgb_to_lab(_random_rgb(128, 128, seed=1)))
    assert lab_n[..., 0].min() >= -1.0 and lab_n[..., 0].max() <= 1.0
    assert lab_n[..., 1:].min() >= -1.0 and lab_n[..., 1:].max() <= 1.0


def test_black_and_white_lightness():
    black = np.zeros((4, 4, 3), np.float32)
    white = np.ones((4, 4, 3), np.float32)
    assert abs(rgb_to_lab(black)[..., 0].max() - 0.0) < 1e-3
    assert abs(rgb_to_lab(white)[..., 0].min() - 100.0) < 1e-2


def test_gray_has_no_color():
    gray = np.full((8, 8, 3), 0.5, np.float32)
    _, ab = rgb_to_model_input_target(gray)
    assert np.abs(ab).max() < 0.02


def test_input_target_shapes():
    L, ab = rgb_to_model_input_target(_random_rgb(32, 48))
    assert L.shape == (32, 48, 1)
    assert ab.shape == (32, 48, 2)


def test_full_pipeline_roundtrip():
    rgb = _random_rgb()
    L, ab = rgb_to_model_input_target(rgb)
    back = model_output_to_rgb(L, ab)
    assert np.abs(rgb - back).max() < 5e-3
