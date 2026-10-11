#!/usr/bin/env bash
# Downloads DAVIS, builds the manifests and splits, and checks the result.
# Run from anywhere; needs the venv active, plus wget and unzip.
set -e
cd "$(dirname "$0")/.."

mkdir -p data/external
if [ ! -d data/external/DAVIS ]; then
  wget -P data/external https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
  unzip -q data/external/DAVIS-2017-trainval-480p.zip -d data/external
  rm data/external/DAVIS-2017-trainval-480p.zip
fi

python -m src.cvpipe.import_frames data/external/DAVIS/JPEGImages/480p
python -m src.cvpipe.apply_splits configs/splits_davis.json data/processed/frames/davis_manifest.parquet
python -m scripts.verify_setup
