import torch
from diffusers import StableDiffusionPipeline
import os

model_id = "runwayml/stable-diffusion-v1-5"
output_path = "./outputs/"
prompt = "A cute 3d render of a kitten"
filename = "cute_3d_render_kitten.png"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    result = pipe(prompt)
    image = result.images[0]

    os.makedirs(output_path, exist_ok=True)
    image.save(os.path.join(output_path, filename))
    print(f"Generated image saved to {os.path.join(output_path, filename)}")

if __name__ == "__main__":
    main()