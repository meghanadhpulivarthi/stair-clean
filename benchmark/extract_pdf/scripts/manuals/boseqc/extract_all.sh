#!/bin/sh
DATASET_ID="manuals/boseqc"
echo "Extracting PDF for dataset ID: ${DATASET_ID}"
bash scripts/pdf/extract.sh "${DATASET_ID}"
bash scripts/tokens/stats.sh "${DATASET_ID}"
bash scripts/chunk/split_all.sh "${DATASET_ID}" 3.45