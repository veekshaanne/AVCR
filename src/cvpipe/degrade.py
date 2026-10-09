"""Old-film degradations, applied to the L channel only.

Input/output: normalized L, float32 (H, W, 1) in [-1, 1] (the dataset contract).
"""
import cv2
import numpy as np


def _to01(L):
    return ((L[..., 0] + 1.0) / 2.0).astype(np.float32)


def _from01(x):
    return (np.clip(x, 0.0, 1.0) * 2.0 - 1.0)[..., None].astype(np.float32)


def _downscale(x, rng):
    f = rng.uniform(0.35, 0.7)
    h, w = x.shape
    small = cv2.resize(x, (max(8, int(w * f)), max(8, int(h * f))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _blur(x, rng):
    return cv2.GaussianBlur(x, (0, 0), rng.uniform(0.6, 2.0))


def _fade(x, rng):
    lo, hi = rng.uniform(0.0, 0.2), rng.uniform(0.75, 1.0)
    return lo + x * (hi - lo)


def _grain(x, rng):
    s = rng.uniform(0.03, 0.10)
    n = rng.normal(0.0, s, x.shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.6)
    weight = 1.0 - 0.6 * np.abs(2.0 * x - 1.0)  # strongest in midtones
    return x + n * weight


def _flicker(x, rng):
    return x * rng.uniform(0.75, 1.25) + rng.uniform(-0.08, 0.08)


def _scratches(x, rng):
    h, w = x.shape
    out = x.copy()
    for _ in range(int(rng.integers(1, 6))):
        c = int(rng.integers(0, w))
        y0, y1 = int(rng.integers(0, h // 2)), int(rng.integers(h // 2, h))
        drift = int(rng.integers(-4, 5))
        val = float(rng.choice([0.0, 1.0]))  # dark or bright scratch
        layer = np.zeros_like(x)
        cv2.line(layer, (c, y0), (c + drift, y1), 1.0, 1)
        layer = cv2.GaussianBlur(layer, (0, 0), 0.6)
        a = rng.uniform(0.4, 0.9) * layer
        out = out * (1.0 - a) + val * a
    return out


def _dust(x, rng):
    h, w = x.shape
    layer = np.zeros_like(x)
    for _ in range(int(rng.integers(5, 40))):
        cv2.circle(layer, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                   int(rng.integers(1, 4)), 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
    a = np.clip(layer * rng.uniform(0.5, 1.0), 0.0, 1.0)
    val = 1.0 if rng.random() < 0.5 else 0.0
    return x * (1.0 - a) + val * a


def _jpeg(x, rng):
    q = int(rng.integers(15, 60))
    u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


EFFECTS = {"downscale": _downscale, "blur": _blur, "fade": _fade,
           "grain": _grain, "flicker": _flicker, "scratches": _scratches,
           "dust": _dust, "jpeg": _jpeg}
ORDER = ["downscale", "blur", "fade", "grain", "flicker",
         "scratches", "dust", "jpeg"]
DEFAULT_PROBS = {"downscale": 0.3, "blur": 0.4, "fade": 0.4, "grain": 0.7,
                 "flicker": 0.5, "scratches": 0.3, "dust": 0.3, "jpeg": 0.4}


class Degrader:
    def __init__(self, probs=None, seed=None):
        self.probs = {**DEFAULT_PROBS, **(probs or {})}
        self.rng = np.random.default_rng(seed)

    def __call__(self, L, rng=None):
        rng = rng or self.rng
        x = _to01(L)
        for name in ORDER:
            if rng.random() < self.probs[name]:
                x = np.clip(EFFECTS[name](x, rng), 0.0, 1.0).astype(np.float32)
        return _from01(x)
