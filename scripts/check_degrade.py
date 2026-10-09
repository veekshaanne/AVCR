import cv2
import numpy as np

from src.cvpipe.color_utils import rgb_to_model_input_target
from src.cvpipe.degrade import Degrader, ORDER

cap = cv2.VideoCapture("data/raw/test.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ok, bgr = cap.read()
cap.release()
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
rgb = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
L, _ = rgb_to_model_input_target(rgb)


def panel(L, text):
    g = (np.clip((L[..., 0] + 1.0) / 2.0, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1, cv2.LINE_AA)
    return img


panels = [panel(L, "original L")]
for name in ORDER:
    d = Degrader(probs={n: (1.0 if n == name else 0.0) for n in ORDER}, seed=1)
    out = d(L)
    print(f"{name:10s} mean abs change: {np.abs(out - L).mean():.4f}")
    panels.append(panel(out, name))
for s in (11, 12, 13):
    panels.append(panel(Degrader(seed=s)(L), f"random mix {s}"))

rows = [np.concatenate(panels[i:i + 4], axis=1) for i in range(0, 12, 4)]
cv2.imwrite("outputs/degrade_check.png", np.concatenate(rows, axis=0))
print("saved outputs/degrade_check.png")
