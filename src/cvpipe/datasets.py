import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.cvpipe.color_utils import rgb_to_model_input_target
from src.cvpipe.degrade import Degrader


class FrameDataset(Dataset):
    """Single-frame dataset for the U-Net baseline.

    degrade_prob: chance (0..1) that a sample's L channel gets old-film
    damage. The ab target is always taken from the clean frame.

    Returns:
        L    : float32 tensor (1, H, W), range [-1, 1]
        ab   : float32 tensor (2, H, W), range ~[-1, 1]
        meta : dict with video_id, scene_id, frame_idx
    """

    def __init__(self, manifest_paths, train=True,
                 resize_short_side=256, crop_size=256, degrade_prob=0.0):
        if isinstance(manifest_paths, str):
            manifest_paths = [manifest_paths]
        self.df = pd.concat([pd.read_parquet(p) for p in manifest_paths],
                            ignore_index=True)
        self.train = train
        self.short = resize_short_side
        self.crop = crop_size
        self.degrade_prob = degrade_prob
        self.degrader = Degrader() if degrade_prob > 0 else None

    def __len__(self):
        return len(self.df)

    def _resize(self, rgb):
        h, w = rgb.shape[:2]
        scale = self.short / min(h, w)
        nh, nw = max(self.crop, round(h * scale)), max(self.crop, round(w * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        return cv2.resize(rgb, (nw, nh), interpolation=interp)

    def _crop(self, rgb):
        h, w = rgb.shape[:2]
        c = self.crop
        if self.train:
            y, x = random.randint(0, h - c), random.randint(0, w - c)
        else:
            y, x = (h - c) // 2, (w - c) // 2
        return rgb[y:y + c, x:x + c]

    def __getitem__(self, i):
        row = self.df.iloc[i]
        bgr = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(row["path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = self._crop(self._resize(rgb))
        L, ab = rgb_to_model_input_target(rgb)

        if self.degrader is not None:
            rng = np.random.default_rng()  # fresh randomness per sample
            if rng.random() < self.degrade_prob:
                L = self.degrader(L, rng=rng)

        meta = {"video_id": row["video_id"],
                "scene_id": int(row["scene_id"]),
                "frame_idx": int(row["frame_idx"])}
        return (torch.from_numpy(L).permute(2, 0, 1).contiguous(),
                torch.from_numpy(ab).permute(2, 0, 1).contiguous(),
                meta)
