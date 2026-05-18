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

echo "Generating a fresh snapshot of installed custom nodes..."
# Execute the snapshot generation using the system python inside ComfyUI's directory
python custom_nodes/ComfyUI-Manager/cm-cli.py save-snapshot

echo "Packing up ComfyUI custom configurations..."

# 2. Create a clean archive of just the essentials
# We include custom_nodes/ but explicitly exclude everything in it EXCEPT the snapshots folder.
# We also forcefully block hidden model extensions (*.safetensors, *.ckpt, *.bin) just in case.
tar -cf - \
    --exclude='models/*' \
    --exclude='venv/*' \
    --exclude='*.safetensors' \
    --exclude='*.ckpt' \
    --exclude='*.bin' \
    --exclude='custom_nodes/*/*' \
    --exclude='custom_nodes/*' \
    --anchored \
    custom_nodes/ComfyUI-Manager/snapshots/ \
    output/ \
    input/ \
    user/ | pv -s $(du -sb output/ input/ user/ custom_nodes/ComfyUI-Manager/snapshots/ 2>/dev/null | awk '{total += $1} END {print total}') | gzip > /workspace/runpod-slim/comfy_essential_backup.tar.gz
    
echo ""
echo "Backup complete!"
echo "➡️ Saved Archive: /workspace/runpod-slim/comfy_essential_backup.tar.gz"
if [ -f "/workspace/runpod-slim/comfy_models_manifest.txt" ]; then
    echo "➡️ Saved Manifest: /workspace/runpod-slim/comfy_models_manifest.txt"
fi
echo "You can now safely download these files via the File Browser UI."