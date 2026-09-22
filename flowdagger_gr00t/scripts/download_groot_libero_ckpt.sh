#!/usr/bin/env bash
# Download the GR00T-N1.7-LIBERO libero_10 checkpoint from the NVIDIA HuggingFace
# release into ${GROOT_CKPT:-~/checkpoints/GR00T-N1.7-LIBERO/libero_10}.
#
# The repo nvidia/GR00T-N1.7-LIBERO holds one subfolder per LIBERO suite; we
# only need libero_10 (the checkpoint this backend uses for task 57).
#
# Requires the Hugging Face CLI provided by huggingface_hub.
set -euo pipefail

GROOT_CKPT="${GROOT_CKPT:-$HOME/checkpoints/GR00T-N1.7-LIBERO/libero_10}"
DEST_ROOT="$(dirname "$GROOT_CKPT")"

echo "Downloading nvidia/GR00T-N1.7-LIBERO (libero_10 subfolder) ..."
echo "  destination: $GROOT_CKPT"
mkdir -p "$DEST_ROOT"

hf download nvidia/GR00T-N1.7-LIBERO \
    --include "libero_10/config.json" "libero_10/embodiment_id.json" \
    "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" \
    "libero_10/processor_config.json" "libero_10/statistics.json" \
    --local-dir "$DEST_ROOT"

echo "Done. Checkpoint at: $GROOT_CKPT"
echo "Point the trainer at it with:  export GROOT_CKPT=$GROOT_CKPT"
