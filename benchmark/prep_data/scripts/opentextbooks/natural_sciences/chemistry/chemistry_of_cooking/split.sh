#!/bin/sh
DATASET_ID="opentextbooks/natural_sciences/chemistry/chemistry_of_cooking"
SECTIONS_JSONL="${HOME}/data/ki_data/cpt/${DATASET_ID}/sections.jsonl"

SINGLE_OUT_JSONL="${HOME}/data/ki_data/cpt/${DATASET_ID}/chunks.all.prefix.jsonl"
for MAX_TOKENS in 256 512 1024 2048; do
    echo "Processing ${MAX_TOKENS} tokens with ${CHUNK_OVERLAP} chunk overlap"
    OUT_JSONL="${HOME}/data/ki_data/cpt/${DATASET_ID}/chunks.${MAX_TOKENS}.prefix.jsonl"
    uv run src/split.py \
      --input_jsonl "${SECTIONS_JSONL}" \
      --output_jsonl "${OUT_JSONL}" \
      --max_tokens ${MAX_TOKENS} \
      --token_to_char 4 \
      --chunk_overlap 32 \
      --add_prefix

    cat "${OUT_JSONL}" >> "${SINGLE_OUT_JSONL}"
done

SINGLE_OUT_JSONL="${HOME}/data/ki_data/cpt/${DATASET_ID}/chunks.all.jsonl"
for MAX_TOKENS in 256 512 1024 2048; do
    echo "Processing ${MAX_TOKENS} tokens with ${CHUNK_OVERLAP} chunk overlap"
    OUT_JSONL="${HOME}/data/ki_data/cpt/${DATASET_ID}/chunks.${MAX_TOKENS}.jsonl"
    uv run src/split.py \
      --input_jsonl "${SECTIONS_JSONL}" \
      --output_jsonl "${OUT_JSONL}" \
      --max_tokens ${MAX_TOKENS} \
      --token_to_char 4 \
      --chunk_overlap 32

    cat "${OUT_JSONL}" >> "${SINGLE_OUT_JSONL}"
done