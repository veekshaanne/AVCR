import sys
import cv2
import numpy as np

from src.cvpipe.color_utils import (
    rgb_to_model_input_target, model_output_to_rgb, lab_to_rgb,
)

video = sys.argv[1] if len(sys.argv) > 1 else "data/raw/test.mp4"
frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 100

cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ok, bgr = cap.read()
cap.release()
if not ok:
    raise SystemExit(f"Could not read frame {frame_idx} from {video}")

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
L, ab = rgb_to_model_input_target(rgb)

# Panel 2: L only, as grayscale
gray = np.repeat((L + 1.0) / 2.0, 3, axis=-1)

# Panel 3: ab only, at constant mid brightness (L=60)
lab_ab = np.concatenate([np.full_like(L, 60.0 / 50.0 - 1.0), ab], axis=-1)
ab_only = model_output_to_rgb(lab_ab[..., :1], lab_ab[..., 1:])

# Panel 4: rebuilt from L + ab
rebuilt = model_output_to_rgb(L, ab)

def label(img, text):
    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    cv2.putText(img8, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img8

panels = [label(rgb, "1 original"), label(gray, "2 L only (input)"),
          label(ab_only, "3 ab only (target)"), label(rebuilt, "4 rebuilt")]
out = np.concatenate(panels, axis=1)

cv2.imwrite("outputs/color_check.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print("saved outputs/color_check.png")
