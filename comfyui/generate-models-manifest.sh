#!/bin/bash

COMFYUI_ROOT="/workspace/runpod-slim/ComfyUI"
MODEL_DIR="$COMFYUI_ROOT/models"
OUTPUT_FILE="/workspace/runpod-slim/comfy_models_manifest.txt"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Directory '$MODEL_DIR' not found."
    exit 1
fi

echo "# ComfyUI Models Manifest with Smart Contextual Search Links" > "$OUTPUT_FILE"
echo "# Format: absolute_path_to_subfolder | filename | HF_search_link" >> "$OUTPUT_FILE"
echo "# --------------------------------------------------------------------------------" >> "$OUTPUT_FILE"

# Find major weight files, ignoring hidden/cache junk
find "$MODEL_DIR" -type f -size +10M ! -path '*/.*' \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pth" -o -name "*.ckpt" \) | while read -r filepath; do
    
    filename=$(basename "$filepath")
    
    # Skip placeholder empty files
    if [[ "$filename" =~ ^put_.*_here$ ]]; then
        continue
    fi

    dir_path=$(dirname "$filepath")
    
    # Extract the relative path underneath the main 'models/' directory to capture folder context
    # Example: /workspace/runpod-slim/ComfyUI/models/RMBG/RMBG-2.0 -> RMBG/RMBG-2.0
    relative_subdirs=${dir_path#*"$MODEL_DIR/"}
    
    # Convert slashes into spaces so it acts as multi-term keywords for HF search
    # Example: RMBG/RMBG-2.0 -> RMBG RMBG-2.0
    folder_context=$(echo "$relative_subdirs" | tr '/' ' ')

    # Define standard top-level Comfy folders that we DON'T need to include in search keywords
    # (Searching "checkpoints" or "diffusion_models" just adds noise to the HF search engine)
    case "$folder_context" in
        checkpoints|loras|vae|embeddings|diffusion_models|text_encoders|clip|unet|controlnet|upscale_models)
            search_query="$filename"
            ;;
        *)
            # For non-standard or nested folders (like florence2, RMBG), mix the folder path with the filename
            search_query="$folder_context $filename"
            ;;
    esac

    # URL encode the spaces and special characters for a browser link
    encoded_query=$(echo "$search_query" | sed 's/ /%20/g; s/\+/%2B/g; s/\&/%26/g')
    
    # Generate the refined Hugging Face search URL
    search_url="https://huggingface.co/models?search=$encoded_query"
    
    echo "$dir_path | $filename | $search_url" >> "$OUTPUT_FILE"
done

echo "Done! Contextual manifest with smart links created at: $OUTPUT_FILE"