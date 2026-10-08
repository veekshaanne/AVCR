"""Inference-side I/O: original frame -> model input, model output -> RGB frame.

Padding convention: reflect padding is added on the RIGHT and BOTTOM only,
so cropping the model output is just ab[:rh, :rw].
"""
import cv2
import numpy as np

from src.cvpipe.color_utils import rgb_to_lab, lab_to_rgb


def prepare_input(rgb: np.ndarray, resize_wh, padded_wh) -> np.ndarray:
    """Original RGB frame (uint8 or float [0,1]) -> normalized L.

    Returns float32 array (H_pad, W_pad, 1) in [-1, 1].
    """
    rw, rh = resize_wh
    pw, ph = padded_wh
    L = rgb_to_lab(rgb)[..., 0] / 50.0 - 1.0
    interp = cv2.INTER_AREA if rw < L.shape[1] else cv2.INTER_CUBIC
    L = cv2.resize(L, (rw, rh), interpolation=interp)
    L = cv2.copyMakeBorder(L, 0, ph - rh, 0, pw - rw, cv2.BORDER_REFLECT_101)
    return L[..., None].astype(np.float32)


def reconstruct_frame(orig_rgb: np.ndarray, ab_pred: np.ndarray, resize_wh) -> np.ndarray:
    """Model ab output + original frame -> RGB float32 in [0,1] at original size.

    ab_pred: float32 (H_pad, W_pad, 2), normalized to [-1, 1].
    If the model gives a torch tensor (2, H, W), convert first with
    ab_pred.permute(1, 2, 0).cpu().numpy().
    """
    h, w = orig_rgb.shape[:2]
    rw, rh = resize_wh
    ab = ab_pred[:rh, :rw]                                   # drop padding
    ab = cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC)
    ab = np.clip(ab, -1.0, 1.0) * 128.0
    L = rgb_to_lab(orig_rgb)[..., :1]                        # original full-res L
    return lab_to_rgb(np.concatenate([L, ab], axis=-1))
