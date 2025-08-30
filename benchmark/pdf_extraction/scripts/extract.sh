#!/bin/sh
OUT_DIR="test"
uv run src/extract.py \
  --pdf_path "${1}" \
  --out_dir "${OUT_DIR}"