import cv2
import numpy as np
import pandas as pd

from src.cvpipe.clips import ClipDataset
from src.cvpipe.color_utils import model_output_to_rgb
from src.cvpipe.datasets import FrameDataset

m = "data/processed/frames/davis_manifest.parquet"
df = pd.read_parquet(m)

clean = FrameDataset(m, train=False, degrade_prob=0.0)
dirty = FrameDataset(m, train=False, degrade_prob=1.0)
print("FrameDataset size :", len(clean), "(manifest rows:", len(df), ")")

clips = ClipDataset(m, clip_len=5, stride=1, train=False)
expected = int(sum(max(0, n - 4) for n in df.groupby("video_id").size()))
print("ClipDataset size  :", len(clips), "| expected:", expected)

L, ab, meta = clips[0]
print("clip shapes       :", tuple(L.shape), tuple(ab.shape))
print("clip frame_idx    :", meta["frame_idx"].tolist(), "| video:", meta["video_id"])

idxs = np.linspace(0, len(clean) - 1, 4).astype(int)
top, bottom = [], []
for i in idxs:
    Lc, abc, mt = clean[i]
    Ld, _, _ = dirty[i]
    rgb = model_output_to_rgb(Lc.permute(1, 2, 0).numpy(), abc.permute(1, 2, 0).numpy())
    top.append(cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    g = (np.clip((Ld[0].numpy() + 1) / 2, 0, 1) * 255).astype(np.uint8)
    bottom.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    print("sample", i, "->", mt["video_id"], "frame", mt["frame_idx"])

cv2.imwrite("outputs/davis_check.png",
            np.concatenate([np.concatenate(top, 1), np.concatenate(bottom, 1)], 0))
print("saved outputs/davis_check.png (top: color from dataset, bottom: degraded L input)")
