import torch
from diffusers import StableDiffusionPipeline
import os

def main():
    # We'll use Stable Diffusion v1.5 as it's standard and relatively fast
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use float16 on GPU to save memory, float32 on CPU
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading Stable Diffusion model ({model_id}).")
    print("Note: If this is the first time running, it will download the model weights (a few GBs).")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe = pipe.to(device)
    except Exception as e:
        print(f"Error loading the model: {e}")
        print("Make sure you have an active internet connection to download the model.")
        return
        
    prompt = "A very cute fluffy kitten looking directly at the camera, highly detailed, photorealistic, 8k resolution"
    print(f"\nGenerating image for prompt: '{prompt}'")
    print("Please wait, generating...")
    
    # Generate the image
    result = pipe(prompt)
    image = result.images[0]
    
    # Save the image to the current directory
    filename = "cute_cat.png"
    output_path = os.path.join(os.getcwd(), filename)
    image.save(output_path)
    print(f"Saved image to: {output_path}")
    
    # Show the image using the default system image viewer
    print("Opening image...")
    image.show()

if __name__ == "__main__":
    main()
