import sys
from pathlib import Path

import pandas as pd

from src.cvpipe.extract import extract_frames
from src.cvpipe.ingest import ingest
from src.cvpipe.normalize import normalize_video

EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def run_batch(input_dir, out_root="data/processed/frames", force=False):
    videos = sorted(p for p in Path(input_dir).iterdir()
                    if p.suffix.lower() in EXTS)
    if not videos:
        raise SystemExit(f"No videos found in {input_dir}")

    manifests, failed = [], []
    for v in videos:
        vid = v.stem
        mpath = Path(out_root) / vid / "manifest.parquet"
        if mpath.exists() and not force:
            print(f"[skip] {vid}: already extracted")
            manifests.append(mpath)
            continue
        try:
            ingest(str(v), "data/interim")                 # video_meta.json + audio
            src, steps = normalize_video(str(v), force=force)
            df = extract_frames(src, out_root=out_root)
            print(f"[ok]   {vid}: {len(df)} frames, "
                  f"{df['scene_id'].nunique()} scenes, fixes: {steps or 'none'}")
            manifests.append(mpath)
        except Exception as e:
            print(f"[FAIL] {vid}: {e}")
            failed.append(vid)

    if not manifests:
        raise SystemExit("Nothing was extracted successfully")
    combined = pd.concat([pd.read_parquet(m) for m in manifests], ignore_index=True)
    out = Path(out_root) / "all_manifest.parquet"
    combined.to_parquet(out, index=False)
    print(f"\ncombined manifest: {out}")
    print(f"videos: {combined['video_id'].nunique()} | frames: {len(combined)} "
          f"| failed: {failed or 'none'}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--force"]
    run_batch(args[0] if args else "data/raw", force="--force" in sys.argv)
