import torch
from diffusers import DiffusionPipeline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
HF_TOKEN = os.environ.get("HF_TOKEN")

model_id = "black-forest-labs/FLUX.2-klein-4B"
output_path = "./outputs/"
prompt = "A cute 3d render of a kitten, highly detailed, 8k resolution"
filename = "cute_flux_kitten.png"

def main():
    print(f"Loading pipeline for {model_id} without quantization...")
    
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"Error loading the model: {e}")
        print("If you run into auth issues, run 'huggingface-cli login'.")
        return

    # Enable CPU offloading. This ensures it fits comfortably in VRAM
    # even without 4-bit quantization, by moving components back to CPU when not in use.
    # pipe.enable_model_cpu_offload()

    print(f"\nGenerating image for prompt: '{prompt}'")
    print("Please wait, generating (Flux might take a few moments)...")
    
    # Flux models (especially distilled ones like schnell/klein) require fewer steps
    result = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=0.0, # Distilled models often use 0.0 guidance
        num_inference_steps=4,
        max_sequence_length=256
    )
    
    image = result.images[0]
    
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, filename)
    image.save(full_path)
    print(f"Saved image to: {full_path}")

if __name__ == "__main__":
    main()
