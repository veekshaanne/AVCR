import cv2
import numpy as np


def iter_frame_chunks(video_path: str, chunk_size: int = 16,
                      start: int = 0, end: int = None):
    """Yield (frame_indices, frames) in chunks.

    frames: float32 RGB array (N, H, W, 3) in [0, 1].
    end is exclusive. Only chunk_size frames are in memory at a time.
    """
    cap = cv2.VideoCapture(str(video_path))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx, buf_i, buf_f = start, [], []
    while end is None or idx < end:
        ok, bgr = cap.read()
        if not ok:
            break
        buf_i.append(idx)
        buf_f.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        idx += 1
        if len(buf_i) == chunk_size:
            yield buf_i, np.stack(buf_f)
            buf_i, buf_f = [], []
    if buf_i:
        yield buf_i, np.stack(buf_f)
    cap.release()
