import json
import math
import subprocess
from pathlib import Path


def probe(path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _fps(rate: str) -> float:
    n, d = rate.split("/")
    return float(n) / float(d) if float(d) else 0.0


def build_meta(path: str, short_side: int = 256, pad_multiple: int = 32) -> dict:
    info = probe(path)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise ValueError(f"No video stream in {path}")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])

    w, h = int(v["width"]), int(v["height"])
    avg, real = _fps(v["avg_frame_rate"]), _fps(v["r_frame_rate"])
    scale = short_side / min(w, h)
    rw, rh = round(w * scale), round(h * scale)
    pw = math.ceil(rw / pad_multiple) * pad_multiple
    ph = math.ceil(rh / pad_multiple) * pad_multiple

    return {
        "source": str(path),
        "width": w,
        "height": h,
        "fps": avg,
        "is_vfr": abs(avg - real) > 0.01,
        "duration": float(info["format"].get("duration", 0)),
        "n_frames": int(v.get("nb_frames", 0)) or None,
        "codec": v["codec_name"],
        "pix_fmt": v["pix_fmt"],
        "field_order": v.get("field_order", "unknown"),
        "has_audio": has_audio,
        "model_resize": [rw, rh],
        "model_padded": [pw, ph],
    }


def ingest(path: str, out_dir: str, **kw) -> dict:
    out = Path(out_dir) / Path(path).stem
    out.mkdir(parents=True, exist_ok=True)
    meta = build_meta(path, **kw)
    if meta["has_audio"]:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vn",
                        "-acodec", "copy", str(out / "audio.mka")], check=True)
    (out / "video_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    import sys
    print(json.dumps(ingest(sys.argv[1], "data/interim"), indent=2))
