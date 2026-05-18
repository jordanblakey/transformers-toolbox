#!/bin/bash

# Basic Flux txt2image models

# put this in /workspace/runpod-slim/ComfyUI/models/diffusion_models/
wget -P /workspace/runpod-slim/ComfyUI/models/diffusion_models/ https://huggingface.co/silveroxides/FLUX.2-dev-fp8_scaled/resolve/a0fed416aa1ce4fc80d507fe2f684089ab506899/flux-2-klein-9b-fp8mixed.safetensors

# put this in /workspace/runpod-slim/ComfyUI/models/vae/
wget -P /workspace/runpod-slim/ComfyUI/models/vae/ https://huggingface.co/trader20261010/flux2-vae.safetensors/resolve/main/flux2-vae.safetensors

# put this in /workspace/runpod-slim/ComfyUI/models/text_encoders/
wget -P /workspace/runpod-slim/ComfyUI/models/text_encoders/ https://huggingface.co/drbaph/Z-Image-fp8/resolve/main/qwen_3_4b_fp8_mixed.safetensors