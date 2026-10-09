import json
import random
from pathlib import Path

import pandas as pd


def make_splits(manifest_paths, out_dir="data/splits", ratios=(0.8, 0.1, 0.1),
                seed=42, by="auto", force=False):
    out = Path(out_dir)
    if (out / "splits.json").exists() and not force:
        raise SystemExit(f"{out/'splits.json'} already exists; use --force to overwrite")

    df = pd.concat([pd.read_parquet(p) for p in manifest_paths], ignore_index=True)
    n_videos = df["video_id"].nunique()
    if by == "auto":
        by = "video" if n_videos >= 5 else "scene"
        if by == "scene":
            print(f"WARNING: only {n_videos} videos, splitting by SCENE (fine for tests, "
                  "leaky for real data)")

    if by == "video":
        df["unit"] = df["video_id"]
    else:
        df["unit"] = df["video_id"] + "/" + df["scene_id"].astype(str).str.zfill(3)

    units = sorted(df["unit"].unique())
    if len(units) < 3:
        raise SystemExit(f"Need at least 3 units to split, found {len(units)}")
    random.Random(seed).shuffle(units)

    n = len(units)
    n_val = max(1, round(n * ratios[1]))
    n_test = max(1, round(n * ratios[2]))
    split_units = {"test": units[:n_test],
                   "val": units[n_test:n_test + n_val],
                   "train": units[n_test + n_val:]}

    out.mkdir(parents=True, exist_ok=True)
    for name, us in split_units.items():
        sub = df[df["unit"].isin(us)].drop(columns="unit").reset_index(drop=True)
        sub.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name:5s}: {len(us):3d} units, {len(sub):6d} frames")
    (out / "splits.json").write_text(json.dumps(
        {"seed": seed, "by": by, "ratios": list(ratios), "units": split_units}, indent=2))
    return split_units


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a != "--force"]
    make_splits(args, force="--force" in sys.argv)
