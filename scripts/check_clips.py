from collections import Counter

import numpy as np
from torch.utils.data import DataLoader

from src.cvpipe.clips import ClipDataset

m = "data/processed/frames/cuts/manifest.parquet"
ds = ClipDataset(m, clip_len=5, stride=2, train=True, degrade_prob=0.0)
print("number of clips          :", len(ds))
print("clips per scene          :", dict(sorted(Counter(
    int(ds.df.iloc[c[0]]["scene_id"]) for c in ds.clips).items())))

bad = 0
for pos in ds.clips:
    r = ds.df.iloc[pos]
    if r["scene_id"].nunique() != 1 or r["video_id"].nunique() != 1:
        bad += 1
    if not (np.diff(r["frame_idx"].to_numpy()) == 2).all():
        bad += 1
print("bad clips (cut/spacing)  :", bad)

L, ab, meta = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))
print("batch L  shape           :", tuple(L.shape))
print("batch ab shape           :", tuple(ab.shape))
print("batch frame_idx shape    :", tuple(meta["frame_idx"].shape))
print("example frame_idx        :", meta["frame_idx"][0].tolist())

# same crop in every frame? scene 1 is a still picture, so frames should match
i = next(k for k, c in enumerate(ds.clips) if ds.df.iloc[c[0]]["scene_id"] == 1)
Ls, _, _ = ds[i]
print("frame-to-frame L change  :", round((Ls[1:] - Ls[:-1]).abs().mean().item(), 6))
