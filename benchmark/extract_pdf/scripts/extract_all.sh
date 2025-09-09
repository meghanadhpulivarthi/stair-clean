#!/bin/sh
DATASET_ID="${1}"
echo "Extracting PDF for dataset ID: ${DATASET_ID}"
bash scripts/pdf/extract.sh "${DATASET_ID}"
bash scripts/chunk/split.sh "${DATASET_ID}"