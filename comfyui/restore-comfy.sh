#!/bin/bash
# Move to workspace
cd /workspace/runpod-slim/ComfyUI

BACKUP_FILE="/workspace/runpod-slim/comfy_essential_backup.tar.gz"

# Check if the backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found at $BACKUP_FILE"
    exit 1
fi

# Check if pv is installed for progress tracking
if command -v pv &> /dev/null; then
    echo "Restoring ComfyUI essentials with progress bar..."
    # Extract using pv to show data flow
    pv "$BACKUP_FILE" | tar -xzf -
else
    echo "pv not found. Restoring silently (this may take a moment)..."
    tar -xzf "$BACKUP_FILE"
fi

echo "Restore complete! Your custom_nodes, inputs, outputs, and user settings have been updated."