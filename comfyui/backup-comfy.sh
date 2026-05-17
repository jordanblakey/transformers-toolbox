#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_SCRIPT="$SCRIPT_DIR/generate-models-manifest.sh"

# Move to workspace
cd /workspace/runpod-slim/ComfyUI

# Check if pv is installed; if not, install it
if ! command -v pv &> /dev/null; then
    echo "pv not found. Installing..."
    apt update -q && apt install -y -qq pv
else
    echo "pv is already installed. Skipping installation."
fi

# 1. Generate the model manifest first
if [ -f "$MANIFEST_SCRIPT" ]; then
    echo "Generating models manifest blueprint..."
    chmod +x "$MANIFEST_SCRIPT"
    "$MANIFEST_SCRIPT"
else
    echo "⚠️ Warning: $MANIFEST_SCRIPT not found. Skipping manifest generation."
fi

echo "Packing up ComfyUI custom configurations..."

# 2. Create a clean archive of just the essentials
tar -cf - \
    --exclude='models/*' \
    --exclude='venv/*' \
    custom_nodes/ \
    output/ \
    input/ \
    user/ | pv -s $(du -sb custom_nodes/ output/ input/ user/ --exclude='models/*' --exclude='venv/*' | awk '{total += $1} END {print total}') | gzip > /workspace/runpod-slim/comfy_essential_backup.tar.gz
    
echo ""
echo "Backup complete!"
echo "➡️ Saved Archive: /workspace/runpod-slim/comfy_essential_backup.tar.gz"
if [ -f "/workspace/runpod-slim/comfy_models_manifest.txt" ]; then
    echo "➡️ Saved Manifest: /workspace/runpod-slim/comfy_models_manifest.txt"
fi
echo "You can now safely download these files via the File Browser UI."