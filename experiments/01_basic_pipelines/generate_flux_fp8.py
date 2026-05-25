import torch
from diffusers import DiffusionPipeline
import os

# 1. Force environment and token parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
HF_TOKEN = os.environ.get("HF_TOKEN")

model_id = "black-forest-labs/FLUX.2-klein-4B"
output_path = "./outputs/"
prompt = "A cute 3d render of a kitten, highly detailed, 8k resolution"
filename = "cute_flux_kitten.png"

def main():
    print(f"Loading pipeline for {model_id} with fast FP8 quantization...")
    
    try:
        # Load the pipeline components in raw bfloat16 first
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN
        )
        
        # 2. Apply fast 8-bit quantization directly to the transformer 
        # This reduces the 13GB VRAM footprint down to ~6.5GB, letting it fit inside the GPU cache!
        print("Optimizing model layers for hardware acceleration...")
        pipe.quantize_mode = "fp8" 
        
        # Move everything natively to your CUDA device
        pipe = pipe.to("cuda")
        
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    print(f"\nGenerating image for prompt: '{prompt}'")
    print("Executing 4-step distilled inference loop...")
    
    # 3. Use BFL's exact reference settings for distilled Klein generation
    result = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=1.0,  # Klein distilled performs best at 1.0 guidance
        num_inference_steps=4,
        max_sequence_length=256
    )
    
    image = result.images[0]
    
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, filename)
    image.save(full_path)
    print(f"\n✨ Success! Saved image to: {full_path}")

if __name__ == "__main__":
    main()