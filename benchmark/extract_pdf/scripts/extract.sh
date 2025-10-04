#!/bin/sh
DATASET_ID="${1}"
BOOK_TITLE="${2}"

DATA_DIR="${HOME}/data/ki_data"
DOCS_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/sections.jsonl"

PDF_PATH="${DATA_DIR}/raw/${DATASET_ID}/content.pdf"
OUT_DIR="${DATA_DIR}/cpt/${DATASET_ID}"
CONFIG_JSON="${DATA_DIR}/raw/${DATASET_ID}/skip.json"

uv run src/pdf/extract.py \
  --pdf_path "${PDF_PATH}" \
  --out_dir "${OUT_DIR}" \
  --skip_json "${CONFIG_JSON}" \
  --book_title "${BOOK_TITLE}"

# uv run src/tokens/stats.py \
#   --input_jsonl "${DOCS_JSONL}" \
#   --out_dir "${OUT_DIR}" \
#   --model_id "mistralai/Mistral-7B-v0.3"

# TOKEN_TO_CHAR="4"
# MAX_TOKENS="2048"

# OUT_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/chunks.${MAX_TOKENS}.jsonl"
# uv run src/chunk/split.py \
#   --input_jsonl "${DOCS_JSONL}" \
#   --output_jsonl "${OUT_JSONL}" \
#   --max_tokens ${MAX_TOKENS} \
#   --token_to_char ${TOKEN_TO_CHAR} \
#   --chunk_overlap 32