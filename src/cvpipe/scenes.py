import json
from pathlib import Path

from scenedetect import ContentDetector, detect

from src.cvpipe.ingest import build_meta


def detect_scenes(video_path: str, out_dir: str = "data/interim",
                  threshold: float = 27.0, min_scene_len: int = 15) -> list:
    meta = build_meta(video_path)
    fps = meta["fps"]
    n_frames = meta["n_frames"] or round(meta["duration"] * fps)

    raw = detect(str(video_path),
                 ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    spans = [(s.frame_num, e.frame_num) for s, e in raw]
    if not spans:  # no cuts found: the whole video is one scene
        spans = [(0, n_frames)]

    scenes = [{
        "scene_id": i,
        "start_frame": a,
        "end_frame": b,  # exclusive
        "n_frames": b - a,
        "start_time": round(a / fps, 3),
        "end_time": round(b / fps, 3),
    } for i, (a, b) in enumerate(spans)]

    out = Path(out_dir) / Path(video_path).stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "scenes.json").write_text(json.dumps(scenes, indent=2))
    return scenes


if __name__ == "__main__":
    import sys
    for sc in detect_scenes(sys.argv[1]):
        print(sc)
