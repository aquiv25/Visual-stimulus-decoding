#!/bin/bash
# Run the full pipeline using the allen_env conda environment.
# Usage: bash run_all.sh

PYTHON=/opt/miniconda3/envs/allen_env/bin/python
cd "$(dirname "$0")"

echo "=== Step 1: Download data ==="
$PYTHON 01_download_data.py || exit 1

echo ""
echo "=== Step 2: Prepare features ==="
$PYTHON 02_prepare_features.py || exit 1

echo ""
echo "=== Step 3: Train and evaluate decoders ==="
$PYTHON 03_decode_model.py || exit 1

echo ""
echo "=== Step 4: Neuron contribution analysis ==="
$PYTHON 04_neuron_contribution.py || exit 1

echo ""
echo "=== Done. Figures saved in ./figures/ ==="
