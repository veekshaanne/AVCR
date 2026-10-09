import cv2
import numpy as np

from src.cvpipe.datasets import FrameDataset

m = "data/processed/frames/cuts/manifest.parquet"
clean = FrameDataset(m, train=False, degrade_prob=0.0)
dirty = FrameDataset(m, train=False, degrade_prob=1.0)

idxs = [0, 80, 150, 200]
top, bottom = [], []
for i in idxs:
    Lc, abc, _ = clean[i]
    Ld, abd, _ = dirty[i]
    print(f"sample {i:3d} | ab max diff: {(abc - abd).abs().max().item():.6f}"
          f" | L mean abs change: {(Lc - Ld).abs().mean().item():.4f}")
    to_img = lambda t: (np.clip((t[0].numpy() + 1) / 2, 0, 1) * 255).astype(np.uint8)
    top.append(to_img(Lc))
    bottom.append(to_img(Ld))

grid = np.concatenate([np.concatenate(top, axis=1),
                       np.concatenate(bottom, axis=1)], axis=0)
cv2.imwrite("outputs/dataset_degrade_check.png", grid)
print("saved outputs/dataset_degrade_check.png (top: clean L, bottom: degraded L)")
