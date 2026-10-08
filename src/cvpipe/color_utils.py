"""Shared RGB <-> LAB conversion. Everyone on the team should use these.

Convention (normalization mode: minus1_1):
  RGB   : float32, [0, 1], shape (H, W, 3)
  LAB   : float32, L in [0, 100], a/b roughly in [-128, 127]
  Model : L_norm = L / 50 - 1   -> [-1, 1]
          ab_norm = ab / 128    -> about [-1, 1]
"""
import cv2
import numpy as np


def _to_float_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    return rgb.astype(np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB (uint8 or float in [0,1]) -> float32 LAB (L: 0..100, ab: ~-128..127)."""
    rgb = np.clip(_to_float_rgb(rgb), 0.0, 1.0)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """float32 LAB -> float32 RGB in [0,1], clipped to the valid gamut."""
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(rgb, 0.0, 1.0)


def normalize_lab(lab: np.ndarray) -> np.ndarray:
    """Raw LAB -> model range [-1, 1]."""
    out = lab.astype(np.float32).copy()
    out[..., 0] = out[..., 0] / 50.0 - 1.0
    out[..., 1:] = out[..., 1:] / 128.0
    return out


def denormalize_lab(lab_norm: np.ndarray) -> np.ndarray:
    """Model range [-1, 1] -> raw LAB."""
    out = lab_norm.astype(np.float32).copy()
    out[..., 0] = (out[..., 0] + 1.0) * 50.0
    out[..., 1:] = out[..., 1:] * 128.0
    return out


def rgb_to_model_input_target(rgb: np.ndarray):
    """RGB image -> (L, ab), both normalized. L: (H,W,1), ab: (H,W,2)."""
    lab = normalize_lab(rgb_to_lab(rgb))
    return lab[..., :1], lab[..., 1:]


def model_output_to_rgb(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Normalized L (H,W,1) + normalized ab (H,W,2) -> RGB float32 [0,1]."""
    lab_norm = np.concatenate([L, ab], axis=-1)
    return lab_to_rgb(denormalize_lab(lab_norm))
