"""Rebuild split parquet files from a committed split list (splits.json).

Run from the repo root, after the manifests exist locally.
"""
import json
import sys
from pathlib import Path

import pandas as pd


def apply_splits(split_json, manifest_paths, out_dir="data/splits"):
    spec = json.load(open(split_json))
    df = pd.concat([pd.read_parquet(p) for p in manifest_paths], ignore_index=True)
    if spec["by"] == "video":
        df["unit"] = df["video_id"]
    else:
        df["unit"] = df["video_id"] + "/" + df["scene_id"].astype(str).str.zfill(3)

    wanted = {u for us in spec["units"].values() for u in us}
    missing = sorted(wanted - set(df["unit"]))
    if missing:
        raise SystemExit(f"{len(missing)} units in the split file are not in the "
                         f"manifests, e.g. {missing[:3]}")
    extra = set(df["unit"]) - wanted
    if extra:
        print(f"note: {len(extra)} units in the manifests are in no split (ignored)")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, units in spec["units"].items():
        sub = df[df["unit"].isin(units)].drop(columns="unit").reset_index(drop=True)
        sub.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name:5s}: {len(units):3d} units, {len(sub):6d} frames")


if __name__ == "__main__":
    args = sys.argv[1:]
    out_dir = "data/splits"
    if "--out" in args:
        i = args.index("--out")
        out_dir = args[i + 1]
        args = args[:i] + args[i + 2:]
    apply_splits(args[0], args[1:], out_dir)
