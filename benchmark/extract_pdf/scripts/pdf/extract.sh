#!/bin/sh
DATA_DIR="${HOME}/data/ki_data"
DATASET_ID="${1}"

PDF_PATH="${DATA_DIR}/raw/${DATASET_ID}/content.pdf"
OUT_DIR="${DATA_DIR}/cpt/${DATASET_ID}"

uv run src/pdf/extract.py \
  --pdf_path "${PDF_PATH}" \
  --out_dir "${OUT_DIR}" \
  --skip_json "${DATA_DIR}/raw/${DATASET_ID}/skip.json"