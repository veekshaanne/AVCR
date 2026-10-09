# Video Colorization: Data Pipeline

## Prerequisites & System Requirements

- **Operating System:** Linux (Ubuntu 20.04+ recommended) or WSL (Windows Subsystem for Linux)
- **Python:** Version 3.11
- **System Binaries:** `ffmpeg` and `ffprobe` installed and accessible via `PATH`

---

## Installation & Verification

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd colorize-cv

```

2. **Set up virtual environment using `uv`:**
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

```


3. **Verify installation:**
```bash
python -m pytest tests/ -q

```


*Expected output:* `8 passed`

> **Note:** All scripts and modules must be executed from the repository root to ensure relative paths inside Parquet manifests resolve correctly.

---

## Quick Start

### Option A: DAVIS Dataset (Recommended for First Runs)

The DAVIS 2017 480p dataset consists of high-quality color video sequences, ideal for establishing training baselines.

1. **Download and extract DAVIS:**
```bash
mkdir -p data/external && cd data/external
wget [https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip)
unzip -q DAVIS-2017-trainval-480p.zip
cd ../..

```


2. **Generate sequence manifests:**
```bash
python -m src.cvpipe.import_frames data/external/DAVIS/JPEGImages/480p

```


3. **Apply reproducible splits:**
```bash
python -m src.cvpipe.apply_splits configs/splits_davis.json data/processed/frames/davis_manifest.parquet

```


This generates `data/splits/train.parquet` (72 videos), `val.parquet` (9 videos), and `test.parquet` (9 videos).

---

### Option B: Processing Custom Raw Videos

Place video files (`.mp4`, `.mkv`, `.avi`, etc.) inside `data/raw/` and execute the batch ingestion pipeline:

```bash
python -m src.cvpipe.batch data/raw

```

**Automated operations per video:**

1. Probes video attributes via `ffprobe`.
2. Normalizes interlacing and variable frame rates (VFR) to constant frame rates (CFR).
3. Performs scene cut detection.
4. Extracts full-resolution PNG frames into `data/processed/frames/<video_id>/<scene_id>/`.
5. Builds and writes Parquet manifests.

---

## Data Conventions & Specifications

### Color Space & Normalization

All color conversions use `src/cvpipe/color_utils.py`. **Do not use OpenCV’s native 8-bit LAB conversion (`cv2.COLOR_BGR2LAB`)**, which clamps and quantizes values differently.

| Component | Target Format & Range | Shape | Notes |
| --- | --- | --- | --- |
| **RGB Image** | `float32`, range $[0.0, 1.0]$ | $(H, W, 3)$ | Base image representation |
| **$L$ Channel** | `float32`, range $[-1.0, 1.0]$ | $(1, H, W)$ | Normalized as: $(L / 50.0) - 1.0$ |
| **$ab$ Channels** | `float32`, range $\approx [-1.0, 1.0]$ | $(2, H, W)$ | Normalized as: $ab / 128.0$ |

> **Model Input Formatting:** Converting $L$ back to ImageNet standards for pretrained backbones (e.g., repeating across 3 channels, applying Mean/Std normalization for VGG16) must be performed inside the model's `forward()` pass or wrapper, **not** inside dataset loaders.

---

### Manifest Schema

Frame metadata is stored in Apache Parquet format at `data/processed/frames/<video_id>/manifest.parquet` (and aggregated in `data/splits/*.parquet`).

| Column Name | Data Type | Description |
| --- | --- | --- |
| `video_id` | `string` | Unique identifier / video name |
| `scene_id` | `int32` / `string` | Zero-indexed shot/scene number within the video |
| `frame_idx` | `int32` | Sequential frame index within the video |
| `timestamp` | `float64` | Frame presentation timestamp in seconds |
| `width` | `int32` | Original frame width in pixels |
| `height` | `int32` | Original frame height in pixels |
| `path` | `string` | Relative filepath to stored image (`.png` or external `.jpg`) |

---

### Metadata Schema

Stored at `data/interim/<video_id>/video_meta.json`:

* Video spatial dimensions (`width`, `height`)
* Frame rate (`fps`), duration, container/codec details
* Interlacing flags, audio channel info
* Model input parameters: `model_resize` and `model_padded`
* `scenes.json`: Boundaries (start/end frame indices) for detected scenes.

---

## Dataset & DataLoader API

### FrameDataset

For single-frame colorization models.

```python
from torch.utils.data import DataLoader
from src.cvpipe.datasets import FrameDataset

ds = FrameDataset(
    manifest_path="data/splits/train.parquet",
    train=True,           # True: random crops; False: center crops
    degrade_prob=0.7      # 70% chance to apply film degradation (grain, scratches, blur)
)

loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)

for L, ab, meta in loader:
    # L:    torch.Tensor of shape (8, 1, 256, 256), range [-1, 1]
    # ab:   torch.Tensor of shape (8, 2, 256, 256), range approx [-1, 1]
    # meta: dict containing frame paths, video_id, and frame_idx
    ...

```

---

### ClipDataset

For temporal or multi-frame models requiring sequence continuity (e.g., 3D-CNNs, Recurrent Networks, or Transformers).

```python
from torch.utils.data import DataLoader
from src.cvpipe.clips import ClipDataset

ds = ClipDataset(
    manifest_path="data/splits/train.parquet",
    clip_len=5,           # Number of consecutive frames per clip
    stride=1,             # Frame step stride
    train=True,
    degrade_prob=0.7
)

loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)

for L_seq, ab_seq, meta in loader:
    # L_seq:  torch.Tensor of shape (4, 5, 1, 256, 256)
    # ab_seq: torch.Tensor of shape (4, 5, 2, 256, 256)
    ...

```

#### Clip Rules:

* Clips **never** cross scene cuts or video boundaries.
* All frames within a single clip share the exact same spatial crop coordinates to maintain spatial coherence.

---

## Train / Val / Test Split Strategy

To prevent data leakage caused by highly redundant consecutive video frames:

1. **Primary Strategy:** Splits are performed strictly at the **video level** with an **80 / 10 / 10** ratio (Random Seed `42`).
2. **Fallback Strategy:** If the total count of available videos is $< 5$, the splitter automatically falls back to splitting at the **scene level** and issues a log warning.
3. **Reproducibility:** DAVIS split definitions are committed in `configs/splits_davis.json`.

---

## Inference & Frame Reconstruction

To reconstruct colorized videos without losing original resolution or detail, predicted $ab$ channels are upscaled to native resolution and merged with the original, un-degraded $L$ channel.

```python
import numpy as np
import torch
from src.cvpipe.ingest import build_meta
from src.cvpipe.reconstruct import prepare_input, reconstruct_frame

# 1. Probe meta parameters for padding and alignment
meta = build_meta("my_video.mp4")
resize, padded = meta["model_resize"], meta["model_padded"]

# 2. Extract and prepare input tensor from full-res RGB/Grayscale frame
L_input = prepare_input(rgb_frame, resize, padded)  # Returns (H_pad, W_pad, 1) in range [-1, 1]

# 3. Model Forward Pass
# ab_pred shape: (H_pad, W_pad, 2) in range [-1, 1]
ab_pred = model(L_input)

# Convert Torch Tensor to Numpy array if needed
if isinstance(ab_pred, torch.Tensor):
    ab_pred = ab_pred.permute(1, 2, 0).detach().cpu().numpy()

# 4. Reconstruct original resolution colorized frame
rgb_out = reconstruct_frame(rgb_frame, ab_pred, resize)  # Full-resolution RGB image in [0.0, 1.0]

```

> **Streaming Large Videos:** For memory-efficient processing of long-form video files without loading full streams into RAM, use `iter_frame_chunks(video_path, chunk_size)` from `src/cvpipe/stream.py`.

---

## Project Directory & Module Architecture

```
colorize-cv/
├── src/
│   └── cvpipe/
│       ├── ingest.py         # Probe video attributes via ffprobe
│       ├── scenes.py         # Scene cut detection algorithms
│       ├── extract.py        # Frame extraction & parquet manifest creation
│       ├── normalize.py      # Deinterlacing & VFR to CFR stabilization
│       ├── batch.py          # Batch processing runner for raw folders
│       ├── import_frames.py  # Dataset importer for folder-based image structures (e.g., DAVIS)
│       ├── color_utils.py    # Standardized RGB <-> LAB conversions
│       ├── datasets.py       # FrameDataset implementation
│       ├── clips.py          # ClipDataset implementation
│       ├── degrade.py        # Synthetic old-film damage generation
│       ├── splits.py         # Video/scene train-val-test split generation
│       ├── apply_splits.py   # Rebuild splits from JSON specifications
│       ├── reconstruct.py    # Preprocessing and full-res color assembly
│       └── stream.py         # Memory-efficient chunked video stream reader
├── configs/                  # Pipeline configurations (default.yaml, splits_davis.json)
├── scripts/                  # Visual and numeric validation scripts
├── tests/                    # Pytest suite
└── data/                     # Ignored in git (raw, interim, processed, external, splits)

```
---

## Known Limitations

* **Chrominance Resolution:** The color model outputs $ab$ tensors at reduced resolution (typically $\approx 256 \times 256$), which are subsequently bilinearly upscaled to original resolution during frame reconstruction. While original edge sharpness is preserved by using the native high-resolution $L$ channel, subtle color boundaries may exhibit slight softness on high-resolution targets.

```

```