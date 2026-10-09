import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.cvpipe.color_utils import rgb_to_lab
from src.cvpipe.infer import run

V = "data/raw/cuts.mp4"


def read_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


cap = cv2.VideoCapture(V)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ok, bgr = cap.read()
cap.release()
orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

for name in ("gray", "tint"):
    print(f"\n--- model: {name}")
    out_root = f"outputs/infer_{name}"
    run(V, model=name, out_root=out_root, limit=100)
    d = Path(out_root) / "cuts"

    man = pd.read_parquet(d / "manifest.parquet")
    n_files = len(list((d / "frames").glob("*.png")))
    scenes = man.groupby("scene_id").size().to_dict()
    hand = json.load(open(d / "handoff.json"))
    out = read_rgb(d / "frames" / "000050.png")
    spread = float((out.max(-1) - out.min(-1)).mean())
    l_diff = float(np.abs(rgb_to_lab(out)[..., 0] - rgb_to_lab(orig)[..., 0]).mean())

    print("frames in manifest / on disk :", len(man), "/", n_files)
    print("frames per scene             :", scenes)
    print("output size (H, W)           :", out.shape[:2], "| original:", orig.shape[:2])
    print("handoff fps / audio          :", hand["fps"], "/", hand["audio"])
    print("color spread (0 = gray)      :", round(spread, 4))
    print("brightness (L) change        :", round(l_diff, 3))
    print("count + scenes               :",
          "PASS" if len(man) == n_files == 100 and scenes == {0: 72, 1: 28} else "FAIL")
    print("original resolution          :", "PASS" if out.shape == orig.shape else "FAIL")
    if name == "gray":
        print("gray model stays gray        :", "PASS" if spread < 0.02 else "FAIL")
        print("original L preserved         :", "PASS" if l_diff < 1.0 else "FAIL")
    else:
        print("tint adds color              :", "PASS" if spread > 0.05 else "FAIL")
