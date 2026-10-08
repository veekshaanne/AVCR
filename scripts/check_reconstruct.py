import cv2
import numpy as np

from src.cvpipe.ingest import build_meta
from src.cvpipe.color_utils import rgb_to_model_input_target
from src.cvpipe.reconstruct import prepare_input, reconstruct_frame

video, idx = "data/raw/test.mp4", 100
meta = build_meta(video)
rw, rh = meta["model_resize"]
pw, ph = meta["model_padded"]

cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
ok, bgr = cap.read()
cap.release()
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

L_in = prepare_input(rgb, (rw, rh), (pw, ph))
print("model input shape  :", L_in.shape)

# Pretend model: return the true ab at model size, padded like a real output
_, ab_true = rgb_to_model_input_target(rgb)
ab_small = cv2.resize(ab_true, (rw, rh), interpolation=cv2.INTER_AREA)
ab_fake = cv2.copyMakeBorder(ab_small, 0, ph - rh, 0, pw - rw, cv2.BORDER_REFLECT_101)
print("fake model output  :", ab_fake.shape)

out = reconstruct_frame(rgb, ab_fake, (rw, rh))
err = np.abs(out - rgb)
print("reconstructed shape:", out.shape)
print("mean abs error     :", round(float(err.mean()), 4))
print("max abs error      :", round(float(err.max()), 4))

side = np.concatenate([rgb, out], axis=1)
cv2.imwrite("outputs/recon_check.png",
            cv2.cvtColor((side * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
print("saved outputs/recon_check.png (left: original, right: rebuilt)")

# --- where is the error? ---
per_pixel = err.max(axis=-1)
print("pixels with error > 0.1  :", round(float((per_pixel > 0.1).mean()) * 100, 3), "%")
print("pixels with error > 0.5  :", round(float((per_pixel > 0.5).mean()) * 100, 3), "%")
print("99.9th percentile error  :", round(float(np.percentile(per_pixel, 99.9)), 4))
heat = (np.clip(per_pixel * 4, 0, 1) * 255).astype(np.uint8)
cv2.imwrite("outputs/recon_error_map.png", heat)
print("saved outputs/recon_error_map.png (white = bigger error)")
