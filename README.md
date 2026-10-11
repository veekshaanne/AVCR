# Video Colorization: Data Pipeline

The data side of the AI video colorization and restoration project. This code turns
raw videos into training-ready tensors, and turns the model's predicted colors back
into finished frames.

The model, temporal smoothing, super-resolution and evaluation are handled by other
team members. This repo gives them clean, tested inputs and outputs.

## How it works

Old black-and-white video has brightness but no color. Every frame is split into two
parts using the LAB color space:

- **L**: brightness (this is what a black-and-white frame already contains)
- **ab**: color

The model takes **L** and predicts **ab**. For training, modern color videos are used:
the color is removed to make fake black-and-white inputs, and the real color is kept
as the answer. To make those inputs resemble real old film, grain, scratches, dust,
blur and fading are added to **L** (never to the answer).

```
video -> frames -> L (input) + ab (answer) -> model -> predicted ab -> color frames
```

## Requirements

- Ubuntu or WSL (any Linux works)
- Python 3.11
- `ffmpeg` and `ffprobe` on the PATH

## Installation

```bash
git clone <repo-url>
cd colorize-cv
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m pytest tests/ -q
```

The last command should print `8 passed`.

Always run commands from the repo root, because the manifests store relative paths.

## Quick start

### Option A: DAVIS (real color videos, good for first runs)

```bash
sudo apt install -y wget unzip     # if not already installed
bash scripts/setup_davis.sh
```

This downloads DAVIS (about 800 MB), builds the manifests, creates the shared
train/val/test split, and runs a check. When it finishes, `data/splits/` contains
`train.parquet`, `val.parquet` and `test.parquet`.

### Option B: your own videos

Put video files in `data/raw/`, then run:

```bash
python -m src.cvpipe.batch data/raw
```

For each video this reads its details, fixes interlacing and variable frame rate
if needed, finds scene cuts, saves every frame as a PNG, and writes manifests.

## Using the data in training

```python
from torch.utils.data import DataLoader
from src.cvpipe.datasets import FrameDataset

ds = FrameDataset("data/splits/train.parquet", train=True, degrade_prob=0.7)
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)

for L, ab, meta in loader:
    # L:  (8, 1, 256, 256)  input, range [-1, 1]
    # ab: (8, 2, 256, 256)  target, range about [-1, 1]
    ...
```

For models that need several consecutive frames:

```python
from src.cvpipe.clips import ClipDataset

ds = ClipDataset("data/splits/train.parquet", clip_len=5, stride=1,
                 train=True, degrade_prob=0.7)
# each item: L_seq (5, 1, 256, 256), ab_seq (5, 2, 256, 256), meta
```

| Option | Meaning |
|---|---|
| `train` | `True`: random crop. `False`: center crop (use for val/test) |
| `degrade_prob` | Chance (0 to 1) that a sample's L gets old-film damage |
| `clip_len`, `stride` | Frames per clip, and the gap between them |

A clip never crosses a scene cut or a video boundary, and all frames in a clip use
the same crop.

## Data format

### Color conventions

All code must use `src/cvpipe/color_utils.py` for color conversion. OpenCV's 8-bit
LAB should not be used.

| Item | Format |
|---|---|
| RGB image | float32, range [0, 1], shape (H, W, 3) |
| Model input `L` | `L / 50 - 1`, range [-1, 1] |
| Model target `ab` | `ab / 128`, range about [-1, 1] |

Preparing `L` for the pretrained VGG16 (mapping back to [0, 1], repeating to 3
channels, ImageNet normalization) belongs inside the model, not in the dataset.

### Manifest

Each video has `data/processed/frames/<video_id>/manifest.parquet` with one row per
frame:

| Column | Meaning |
|---|---|
| `video_id` | Video name |
| `scene_id` | Shot number within the video |
| `frame_idx` | Frame number in the video |
| `timestamp` | Seconds from the start |
| `width`, `height` | Original frame size |
| `path` | Path to the image file |

Frames from video files are saved as `<video_id>/<scene_id>/<frame_idx>.png` at full
resolution. For DAVIS, nothing is copied: the manifest points to the original JPEGs.

### Video info

`data/interim/<video_id>/video_meta.json` stores the size, fps, duration, codec,
interlacing, audio info, and the model input sizes (`model_resize`, `model_padded`).
`scenes.json` lists where each scene starts and ends.

## For the Model Team

The dataset already provides everything needed for training: a black-and-white input
and the matching true color for every sample. No conversion is required on the model
side.

### What the dataset provides

| Item | Shape | Range | Meaning |
|---|---|---|---|
| `L` | `(batch, 1, 256, 256)` | [-1, 1] | Black-and-white input (brightness) |
| `ab` | `(batch, 2, 256, 256)` | about [-1, 1] | True color, the training target |

### What the model must do

- Take `L` and output a predicted `ab` with the same shape as the target.
- End with a `tanh` activation, so the output stays in [-1, 1].
- Handle the VGG16 input requirements internally (map `L` back to [0, 1], repeat it to
  3 channels, apply ImageNet mean/std). The dataset does not do this.
- Stay fully convolutional. Training uses 256x256 crops, while inference uses full
  frames padded to a multiple of 32.

### Training

```python
from torch.utils.data import DataLoader
from src.cvpipe.datasets import FrameDataset

train = DataLoader(FrameDataset("data/splits/train.parquet", train=True, degrade_prob=0.7),
                   batch_size=8, shuffle=True)
val = DataLoader(FrameDataset("data/splits/val.parquet", train=False), batch_size=8)

for L, ab_true, meta in train:
    ab_pred = model(L)
    loss = (ab_pred - ab_true).abs().mean()    # L1 loss; other losses can be added
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Use `train=False` for validation and test data, which gives a fixed center crop instead of
a random one. For models that need consecutive frames, use `ClipDataset` (see the
Dataset & DataLoader API section).

### Viewing a colorized result

Combine the original brightness with the predicted color:

```python
from src.cvpipe.color_utils import model_output_to_rgb

L, ab_true, _ = next(iter(val))
ab_pred = model(L)
rgb = model_output_to_rgb(L[0].permute(1, 2, 0).numpy(),
                          ab_pred[0].permute(1, 2, 0).detach().numpy())
# rgb: (256, 256, 3), float32, range [0, 1]
```

### Colorizing a real black-and-white video

The helpers in `src/cvpipe/reconstruct.py` handle the resizing, padding and merging.
`iter_frame_chunks` in `src/cvpipe/stream.py` reads a video a few frames at a time.

```python
import torch
from src.cvpipe.ingest import build_meta
from src.cvpipe.reconstruct import prepare_input, reconstruct_frame

meta = build_meta("film.mp4")
resize, padded = meta["model_resize"], meta["model_padded"]

# rgb: one frame, float32 RGB in [0, 1], shape (H, W, 3)
L = prepare_input(rgb, resize, padded)                    # (H_pad, W_pad, 1)
x = torch.from_numpy(L).permute(2, 0, 1)[None]            # (1, 1, H_pad, W_pad)
with torch.no_grad():
    ab = model(x)[0].permute(1, 2, 0).cpu().numpy()       # (H_pad, W_pad, 2)
rgb_out = reconstruct_frame(rgb, ab, resize)              # (H, W, 3), range [0, 1]
```

`reconstruct_frame` removes the padding, enlarges `ab` to the original size, and merges it
with the original full-resolution brightness, so the output keeps the sharpness of the
source frame.

## Train/val/test split

Splits are made by video, never by frame, because neighboring frames look almost
identical and would leak between sets. The seed is 42 and the ratio is 80/10/10.
When fewer than 5 videos exist, the split falls back to scenes and prints a warning
(fine for tests, not for real data).

The DAVIS split list is committed at `configs/splits_davis.json`, so every run uses
the same videos.

## Running a trained model on a video

```python
from src.cvpipe.ingest import build_meta
from src.cvpipe.reconstruct import prepare_input, reconstruct_frame

meta = build_meta("my_video.mp4")
resize, padded = meta["model_resize"], meta["model_padded"]

L = prepare_input(rgb, resize, padded)        # (H_pad, W_pad, 1) in [-1, 1]
ab = model(L)                                  # (H_pad, W_pad, 2) in [-1, 1]
rgb_out = reconstruct_frame(rgb, ab, resize)  # full-size color frame, [0, 1]
```

`reconstruct_frame` removes the padding, enlarges `ab` to the original size, and
combines it with the original full-resolution L, so brightness and sharpness stay as
sharp as the source. If the model returns a torch tensor of shape (2, H, W), convert
it first with `.permute(1, 2, 0).cpu().numpy()`.

For long videos, `iter_frame_chunks(video_path, chunk_size)` in `stream.py` reads a
few frames at a time instead of loading the whole film.

## Project structure

```
src/cvpipe/
  ingest.py        read video details (ffprobe)
  scenes.py        find scene cuts
  extract.py       save frames + manifest
  normalize.py     fix interlaced / variable-fps video
  batch.py         process a whole folder of videos
  import_frames.py import datasets stored as image folders (DAVIS)
  color_utils.py   RGB <-> LAB and normalization
  datasets.py      FrameDataset
  clips.py         ClipDataset
  degrade.py       old-film damage effects
  splits.py        create train/val/test splits
  apply_splits.py  rebuild splits from the committed list
  reconstruct.py   model input prep and output reconstruction
  stream.py        read long videos in chunks
configs/           default.yaml, splits_davis.json
scripts/           check_*.py: quick visual and numeric sanity checks
tests/             pytest tests
data/              not in git (raw, interim, processed, external, splits)
```

## Status

**Done and tested:** video ingest, scene detection, frame extraction, LAB conversion
(8 tests), frame and clip datasets, film degradations, splits, reconstruction,
streaming, DAVIS import.

## Known limitation

The model predicts `ab` at about 256 px, and the result is enlarged afterwards, so
color edges are slightly softer than the sharp original brightness.
