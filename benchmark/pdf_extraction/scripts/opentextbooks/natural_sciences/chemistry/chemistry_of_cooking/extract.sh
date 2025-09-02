#!/bin/sh
DATASET_ID="opentextbooks/natural_sciences/chemistry/chemistry_of_cooking"
DATA_DIR="${HOME}/data/ki_data/raw/${DATASET_ID}"
OUT_DIR="${HOME}/data/ki_data/cpt/${DATASET_ID}"
uv run src/extract.py \
  --pdf_path "${DATA_DIR}/content.pdf" \
  --out_dir "${OUT_DIR}" \
  --book_title "Chemistry of Cooking"