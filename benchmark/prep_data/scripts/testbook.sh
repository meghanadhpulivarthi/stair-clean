#!/bin/sh
CONTENT_JSONL="${HOME}/data/ki_data/raw/opentextbooks/testbook/content.jsonl"
OUT_JSONL="${HOME}/data/ki_data/cpt/opentextbooks/testbook/content.jsonl"

uv run src/split.py \
  --input_jsonl "${CONTENT_JSONL}" \
  --output_jsonl "${OUT_JSONL}" \
  --max_tokens 1024 \
  --token_to_char 4 \
  --chunk_overlap 128 \
  --add_prefix


OUT_JSONL="${HOME}/data/ki_data/cpt/opentextbooks/testbook/chunks.jsonl"
uv run src/split.py \
  --input_jsonl "${CONTENT_JSONL}" \
  --output_jsonl "${OUT_JSONL}" \
  --max_tokens 512 \
  --token_to_char 4 \
  --chunk_overlap 128