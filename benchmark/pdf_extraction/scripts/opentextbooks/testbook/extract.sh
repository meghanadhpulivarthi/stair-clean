#!/bin/sh
DATASET_ID="opentextbooks/testbook"
DATA_DIR="${HOME}/data/ki_data/raw/${DATASET_ID}"
OUT_DIR="${HOME}/data/ki_data/cpt/${DATASET_ID}"
uv run src/extract.py \
  --pdf_path "${DATA_DIR}/content.pdf" \
  --out_dir "${OUT_DIR}"