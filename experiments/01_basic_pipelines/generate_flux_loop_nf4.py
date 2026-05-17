import torch
from diffusers import Flux2KleinPipeline  # <--- The verified working class
from diffusers.quantizers import PipelineQuantizationConfig
import os

# Force environment context
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
HF_TOKEN = os.environ.get("HF_TOKEN")

model_id = "black-forest-labs/FLUX.2-klein-4B"
output_path = "./outputs/"

def initialize_pipeline():
    """Loads the Klein components into VRAM once using the proper class blueprint."""
    print("🔥 Initializing 4-bit Quantization Config...")
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }
    )
    
    print(f"🔥 Force-loading Klein model into VRAM: {model_id}...")
    # Swapped to the correct dedicated pipeline class
    raw_pipe = Flux2KleinPipeline.from_pretrained(
        model_id,
        quantization_config=pipeline_quant_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=HF_TOKEN
    )
    return raw_pipe

def generate_image(pipe, prompt, index):
    """Executes ultra-fast generation loops without re-loading data."""
    print(f"\n🎨 Processing Prompt [{index}]: '{prompt}'")
    
    result = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=1.0, 
        num_inference_steps=4,
        max_sequence_length=256
    )
    
    os.makedirs(output_path, exist_ok=True)
    filename = f"flux_render_{index}.png"
    full_path = os.path.join(output_path, filename)
    
    result.images[0].save(full_path)
    result.images[0].show()
    print(f"✨ Success! Saved fast render to: {full_path}")

def main():
    try:
        # Load once. This takes a moment, but keeps the memory warm.
        pipe = initialize_pipeline()
    except Exception as e:
        print(f"Error during model VRAM loading: {e}")
        return

    print("\n🚀 VRAM IS HOT. Enter your prompts below.")
    print("Type 'exit' or 'quit' to close the program and free your GPU.")
    print("-" * 50)

    prompt_counter = 1
    while True:
        user_prompt = input(f"\n[Prompt {prompt_counter}] Enter text description: ").strip()
        
        if user_prompt.lower() in ['exit', 'quit']:
            print("Cooling down GPU. Exiting application...")
            break
            
        if not user_prompt:
            print("Prompt cannot be empty. Try again!")
            continue
            
        try:
            generate_image(pipe, user_prompt, prompt_counter)
            prompt_counter += 1
        except Exception as e:
            print(f"Generation error occurred: {e}")

if __name__ == "__main__":
    main()