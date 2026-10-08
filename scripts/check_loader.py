import yaml
from torch.utils.data import DataLoader

from src.cvpipe.datasets import FrameDataset

cfg = yaml.safe_load(open("configs/default.yaml"))
manifest = "data/processed/frames/cuts/manifest.parquet"

ds = FrameDataset(manifest, train=True,
                  resize_short_side=cfg["resize_short_side"],
                  crop_size=cfg["crop_size"])
loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

L, ab, meta = next(iter(loader))
print("dataset size :", len(ds))
print("L  shape     :", tuple(L.shape), L.dtype)
print("ab shape     :", tuple(ab.shape), ab.dtype)
print("L  range     :", round(L.min().item(), 3), "to", round(L.max().item(), 3))
print("ab range     :", round(ab.min().item(), 3), "to", round(ab.max().item(), 3))
print("scene_ids    :", meta["scene_id"].tolist())
print("frame_idx    :", meta["frame_idx"].tolist())
