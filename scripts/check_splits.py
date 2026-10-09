import json

import pandas as pd

from src.cvpipe.clips import ClipDataset
from src.cvpipe.datasets import FrameDataset

S = "data/splits"
sp = {k: pd.read_parquet(f"{S}/{k}.parquet") for k in ("train", "val", "test")}
meta = json.load(open(f"{S}/splits.json"))
print("split mode :", meta["by"], "| seed", meta["seed"])

paths = {k: set(v["path"]) for k, v in sp.items()}
print("overlap train/val :", len(paths["train"] & paths["val"]))
print("overlap train/test:", len(paths["train"] & paths["test"]))
print("overlap val/test  :", len(paths["val"] & paths["test"]))
print("total frames      :", sum(len(v) for v in sp.values()))

scene_keys = {k: set(zip(v["video_id"], v["scene_id"])) for k, v in sp.items()}
print("scenes per split  :", {k: len(v) for k, v in scene_keys.items()})
print("scenes split across sets:",
      len(scene_keys["train"] & scene_keys["val"]) +
      len(scene_keys["train"] & scene_keys["test"]) +
      len(scene_keys["val"] & scene_keys["test"]))

tr = FrameDataset(f"{S}/train.parquet", train=True, degrade_prob=0.7)
va = FrameDataset(f"{S}/val.parquet", train=False)
cl = ClipDataset(f"{S}/val.parquet", clip_len=5, train=False)
print("train frames / val frames / val clips:", len(tr), len(va), len(cl))
print("train sample L shape:", tuple(tr[0][0].shape))
