"""Import datasets stored as folders of image frames (e.g. DAVIS).

Each sub-folder of `root` is treated as ONE video with ONE scene.
No re-encoding: manifests point at the original image files.
"""
import sys
from pathlib import Path

import cv2
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def import_image_sequences(root, out_root="data/processed/frames",
                           prefix="davis_", fps=24.0, force=False):
    seq_dirs = sorted(d for d in Path(root).iterdir() if d.is_dir())
    if not seq_dirs:
        raise SystemExit(f"No sequence folders found in {root}")

    manifests, sizes = [], set()
    for d in seq_dirs:
        video_id = f"{prefix}{d.name}"
        mpath = Path(out_root) / video_id / "manifest.parquet"
        if mpath.exists() and not force:
            df = pd.read_parquet(mpath)
        else:
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
            if not files:
                print(f"[skip] {d.name}: no images")
                continue
            first = cv2.imread(str(files[0]))
            if first is None:
                print(f"[FAIL] {d.name}: cannot read {files[0]}")
                continue
            h, w = first.shape[:2]
            df = pd.DataFrame({
                "video_id": video_id,
                "scene_id": 0,
                "frame_idx": range(len(files)),
                "timestamp": [round(i / fps, 4) for i in range(len(files))],
                "width": w,
                "height": h,
                "path": [str(p) for p in files],
            })
            mpath.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(mpath, index=False)
        sizes.add((int(df["width"].iloc[0]), int(df["height"].iloc[0])))
        manifests.append(df)

    combined = pd.concat(manifests, ignore_index=True)
    out = Path(out_root) / f"{prefix}manifest.parquet"
    combined.to_parquet(out, index=False)

    per_seq = combined.groupby("video_id").size()
    print(f"sequences        : {combined['video_id'].nunique()}")
    print(f"total frames     : {len(combined)}")
    print(f"frames per seq   : min {per_seq.min()}, max {per_seq.max()}")
    print(f"resolutions      : {sorted(sizes)}")
    print(f"combined manifest: {out}")
    return combined


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--force"]
    import_image_sequences(args[0], force="--force" in sys.argv)
