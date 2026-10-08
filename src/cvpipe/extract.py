from pathlib import Path

import cv2
import pandas as pd

from src.cvpipe.ingest import build_meta
from src.cvpipe.scenes import detect_scenes


def extract_frames(video_path: str,
                   out_root: str = "data/processed/frames",
                   interim: str = "data/interim") -> pd.DataFrame:
    video_id = Path(video_path).stem
    meta = build_meta(video_path)
    scenes = detect_scenes(video_path, out_dir=interim)
    fps = meta["fps"]

    root = Path(out_root) / video_id
    cap = cv2.VideoCapture(str(video_path))
    rows = []
    for sc in scenes:
        sdir = root / f"{sc['scene_id']:03d}"
        sdir.mkdir(parents=True, exist_ok=True)
        for idx in range(sc["start_frame"], sc["end_frame"]):
            ok, bgr = cap.read()
            if not ok:
                break
            path = sdir / f"{idx:06d}.png"
            cv2.imwrite(str(path), bgr)
            rows.append({
                "video_id": video_id,
                "scene_id": sc["scene_id"],
                "frame_idx": idx,
                "timestamp": round(idx / fps, 4),
                "width": meta["width"],
                "height": meta["height"],
                "path": str(path),
            })
    cap.release()

    df = pd.DataFrame(rows)
    df.to_parquet(root / "manifest.parquet", index=False)
    return df


if __name__ == "__main__":
    import sys
    df = extract_frames(sys.argv[1])
    print(f"saved {len(df)} frames")
    print(df.groupby("scene_id").size())
