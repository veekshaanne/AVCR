"""Check that the data pipeline works. Run from the repo root after the Quick start."""
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.cvpipe.clips import ClipDataset
from src.cvpipe.datasets import FrameDataset

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print("PASS" if ok else "FAIL", "-", name, detail)


S = Path("data/splits")
missing = [n for n in ("train", "val", "test") if not (S / f"{n}.parquet").exists()]
if missing:
    sys.exit(f"Missing split files: {missing}. Run the Quick start first.")

sp = {n: pd.read_parquet(S / f"{n}.parquet") for n in ("train", "val", "test")}
paths = {n: set(df["path"]) for n, df in sp.items()}
overlap = (len(paths["train"] & paths["val"]) + len(paths["train"] & paths["test"])
           + len(paths["val"] & paths["test"]))
check("no frame appears in two splits", overlap == 0)

tr = FrameDataset(str(S / "train.parquet"), train=True, degrade_prob=0.7)
va = FrameDataset(str(S / "val.parquet"), train=False)
cl = ClipDataset(str(S / "val.parquet"), clip_len=5, train=False)
check("datasets are not empty", len(tr) > 0 and len(va) > 0 and len(cl) > 0,
      f"(train {len(tr)}, val {len(va)}, val clips {len(cl)})")

bs = 8
L, ab, _ = next(iter(DataLoader(tr, batch_size=bs, shuffle=True)))
check("frame batch shapes",
      tuple(L.shape) == (bs, 1, 256, 256) and tuple(ab.shape) == (bs, 2, 256, 256),
      f"{tuple(L.shape)} {tuple(ab.shape)}")
check("values stay within [-1, 1]",
      min(L.min().item(), ab.min().item()) >= -1.0
      and max(L.max().item(), ab.max().item()) <= 1.0)

Ls, abs_, _ = cl[0]
check("clip shapes",
      tuple(Ls.shape) == (5, 1, 256, 256) and tuple(abs_.shape) == (5, 2, 256, 256),
      f"{tuple(Ls.shape)} {tuple(abs_.shape)}")

per_scene = sp["val"].groupby(["video_id", "scene_id"]).size()
expected = int(sum(max(0, n - 4) for n in per_scene))
check("clips never cross a scene or video boundary", len(cl) == expected,
      f"({len(cl)} clips, expected {expected})")

print("\nAll checks passed." if all(results) else "\nSome checks FAILED.")
sys.exit(0 if all(results) else 1)
