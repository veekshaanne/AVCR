import subprocess
from pathlib import Path

from src.cvpipe.ingest import build_meta

INTERLACED = {"tt", "bb", "tb", "bt"}


def plan_normalization(meta: dict) -> list:
    steps = []
    if meta["field_order"] in INTERLACED:
        steps.append("deinterlace")
    if meta["is_vfr"]:
        steps.append("constant_fps")
    return steps


def normalize_video(video_path, out_dir="data/interim/normalized", force=False):
    """Return (path_to_use, steps_applied). Untouched if no fix is needed."""
    meta = build_meta(str(video_path))
    steps = plan_normalization(meta)
    if not steps:
        return str(video_path), steps
    if meta["fps"] <= 0:
        raise ValueError(f"Cannot read a frame rate from {video_path}")

    out = Path(out_dir) / (Path(video_path).stem + ".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        return str(out), steps

    filters = []
    if "deinterlace" in steps:
        filters.append("yadif=mode=0")           # one output frame per input frame
    if "constant_fps" in steps:
        filters.append(f"fps={meta['fps']:.6f}")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
         "-vf", ",".join(filters), "-an",
         "-c:v", "libx264", "-crf", "10", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", str(out)],
        check=True)
    return str(out), steps
