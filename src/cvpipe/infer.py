"""Run a video through a colorization model and save color frames.

Model interface: model(L_batch) -> ab_batch
  L_batch : float32 (N, H_pad, W_pad, 1), range [-1, 1]
  ab_batch: float32 (N, H_pad, W_pad, 2), range [-1, 1]

Usage:
  python -m src.cvpipe.infer <video> [--model gray|tint] [--limit N]
With a real model:
  from src.cvpipe.infer import run, torch_model_fn
  run("film.mp4", model=torch_model_fn(net, device="cuda"))
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.cvpipe.ingest import ingest
from src.cvpipe.normalize import normalize_video
from src.cvpipe.reconstruct import prepare_input, reconstruct_frame
from src.cvpipe.scenes import detect_scenes
from src.cvpipe.stream import iter_frame_chunks


def _gray_model(L):
    return np.zeros(L.shape[:3] + (2,), np.float32)


def _tint_model(L):
    ab = np.empty(L.shape[:3] + (2,), np.float32)
    ab[..., 0], ab[..., 1] = 0.10, 0.25
    return ab


DUMMY_MODELS = {"gray": _gray_model, "tint": _tint_model}


def torch_model_fn(net, device="cpu"):
    """Wrap a torch model: takes (N,1,H,W) tensor, returns (N,2,H,W) tensor."""
    import torch
    net = net.to(device).eval()

    def fn(L):
        x = torch.from_numpy(L).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            ab = net(x)
        return ab.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    return fn


def run(video_path, model="gray", out_root="outputs", chunk_size=8, limit=None):
    video_path = str(video_path)
    video_id = Path(video_path).stem
    model_fn = DUMMY_MODELS[model] if isinstance(model, str) else model
    model_name = model if isinstance(model, str) else getattr(model, "__name__", "custom")

    meta = ingest(video_path, "data/interim")          # video_meta.json + audio
    src, fixes = normalize_video(video_path)           # no-op for clean video
    if fixes:
        print("old-film fixes applied:", fixes)
    scenes = detect_scenes(src, out_dir="data/interim")
    starts = np.array([s["start_frame"] for s in scenes])
    resize, padded, fps = meta["model_resize"], meta["model_padded"], meta["fps"]

    out_dir = Path(out_root) / video_id
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idxs, frames in iter_frame_chunks(src, chunk_size, end=limit):
        L_batch = np.stack([prepare_input(f, resize, padded) for f in frames])
        ab_batch = model_fn(L_batch)
        for i, f, ab in zip(idxs, frames, ab_batch):
            rgb = reconstruct_frame(f, ab, resize)
            path = frames_dir / f"{i:06d}.png"
            img = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            k = max(0, int(np.searchsorted(starts, i, side="right")) - 1)
            rows.append({"video_id": video_id,
                         "scene_id": int(scenes[k]["scene_id"]),
                         "frame_idx": i,
                         "timestamp": round(i / fps, 4),
                         "path": str(path)})
    if not rows:
        raise SystemExit(f"No frames could be read from {src}")

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "manifest.parquet", index=False)
    audio = Path("data/interim") / video_id / "audio.mka"
    handoff = {
        "video_id": video_id, "source": video_path, "model": model_name,
        "frames_dir": str(frames_dir), "manifest": str(out_dir / "manifest.parquet"),
        "fps": fps, "width": meta["width"], "height": meta["height"],
        "frames_written": len(df), "n_scenes": int(df["scene_id"].nunique()),
        "audio": str(audio) if meta["has_audio"] else None,
        "fixes_applied": fixes,
        "frame_format": "RGB PNG, original resolution, one file per frame",
    }
    (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2))
    print(f"wrote {len(df)} frames, {handoff['n_scenes']} scenes -> {out_dir}")
    return handoff


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--model", default="gray", choices=list(DUMMY_MODELS))
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    run(a.video, model=a.model, out_root=a.out, chunk_size=a.chunk, limit=a.limit)
