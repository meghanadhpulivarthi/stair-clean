#!/bin/sh
DATASET_ID="${1}"
BOOK_TITLE="${2}"

DATA_DIR="${HOME}/data/ki_data"
CONTENT_DIR="${DATA_DIR}/raw/${DATASET_ID}"
OUT_DIR="${DATA_DIR}/dsi/${DATASET_ID}"

PDF_PATH="${CONTENT_DIR}/content.pdf"
CONFIG_JSON="${CONTENT_DIR}/skip.json"

uv run src/pdf/extract.py \
  --pdf_path "${PDF_PATH}" \
  --out_dir "${OUT_DIR}" \
  --skip_json "${CONFIG_JSON}" \
  --book_title "${BOOK_TITLE}"