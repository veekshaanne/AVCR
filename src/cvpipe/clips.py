import random

import cv2
import numpy as np
import torch

from src.cvpipe.color_utils import rgb_to_model_input_target
from src.cvpipe.datasets import FrameDataset


class ClipDataset(FrameDataset):
    """Short sequences of consecutive frames from ONE scene.

    Returns:
        L_seq  : float32 tensor (T, 1, H, W), range [-1, 1]
        ab_seq : float32 tensor (T, 2, H, W), range ~[-1, 1]
        meta   : dict with video_id, scene_id, frame_idx (tensor of length T)
    """

    def __init__(self, manifest_paths, clip_len=5, stride=1, train=True,
                 resize_short_side=256, crop_size=256, degrade_prob=0.0):
        super().__init__(manifest_paths, train=train,
                         resize_short_side=resize_short_side,
                         crop_size=crop_size, degrade_prob=degrade_prob)
        self.clip_len = clip_len
        self.stride = stride
        span = (clip_len - 1) * stride + 1

        self.clips = []  # each item: row positions of the frames in one clip
        for _, g in self.df.groupby(["video_id", "scene_id"]):
            g = g.sort_values("frame_idx")
            rows = g.index.to_numpy()
            fidx = g["frame_idx"].to_numpy()
            for s in range(0, len(rows) - span + 1):
                if fidx[s + span - 1] - fidx[s] != span - 1:
                    continue  # gap in the frames, skip
                self.clips.append(rows[s:s + span:stride])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        rows = self.df.iloc[self.clips[i]]
        frames = []
        for p in rows["path"]:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(p)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(self._resize(rgb))

        # one crop window for the whole clip
        h, w = frames[0].shape[:2]
        c = self.crop
        if self.train:
            y, x = random.randint(0, h - c), random.randint(0, w - c)
        else:
            y, x = (h - c) // 2, (w - c) // 2

        rng = np.random.default_rng()
        do_degrade = (self.degrader is not None
                      and rng.random() < self.degrade_prob)  # decided once per clip

        Ls, abs_ = [], []
        for rgb in frames:
            L, ab = rgb_to_model_input_target(rgb[y:y + c, x:x + c])
            if do_degrade:
                L = self.degrader(L, rng=rng)
            Ls.append(L)
            abs_.append(ab)

        L_seq = torch.from_numpy(np.stack(Ls)).permute(0, 3, 1, 2).contiguous()
        ab_seq = torch.from_numpy(np.stack(abs_)).permute(0, 3, 1, 2).contiguous()
        meta = {"video_id": rows["video_id"].iloc[0],
                "scene_id": int(rows["scene_id"].iloc[0]),
                "frame_idx": torch.tensor(rows["frame_idx"].to_numpy())}
        return L_seq, ab_seq, meta
