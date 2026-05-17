import torch
from diffusers import Flux2KleinPipeline  # <--- Use the exact dedicated class
from diffusers.quantizers import PipelineQuantizationConfig
import os

# Force environment context
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
HF_TOKEN = os.environ.get("HF_TOKEN")

model_id = "black-forest-labs/FLUX.2-klein-4B"
output_path = "./outputs/"
prompt = "A cute 3d render of a kitten, highly detailed, 8k resolution"
filename = "cute_flux_kitten_4bit.png"

def main():
    print(f"Initializing official PipelineQuantizationConfig for bitsandbytes...")
    
    # 4-bit memory packing
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }
    )
    
    try:
        print(f"Loading native unified 4-bit Klein pipeline...")
        # Loading with the native class automatically resolves the component mapping
        pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            quantization_config=pipeline_quant_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            token=HF_TOKEN
        )
        
    except Exception as e:
        print(f"Error during optimized model initialization: {e}")
        return

    print(f"\nGenerating image for prompt: '{prompt}'")
    print("Executing hardware-accelerated local VRAM inference...")
    
    result = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=1.0, # Klein distilled performs best at reference 1.0 guidance
        num_inference_steps=4,
        max_sequence_length=256
    )
    
    image = result.images[0]
    
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, filename)
    image.save(full_path)
    image.show()
    print(f"\n✨ Success! Saved fast 4-bit render to: {full_path}")

if __name__ == "__main__":
    main()