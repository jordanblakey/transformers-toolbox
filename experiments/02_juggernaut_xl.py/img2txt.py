# # import torch
# # from diffusers import DiffusionPipeline

# # pipe = DiffusionPipeline.from_pretrained(
# #     "RunDiffusion/Juggernaut-XL-v9", 
# #     torch_dtype=torch.float16, 
# #     variant="fp16", 
# #     use_safetensors=True
# # ).to("cuda")

# # print(pipe)

# # # prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# # # image = pipe(prompt).images[0]

# from diffusers import StableDiffusionXLPipeline

# # This only downloads the tiny 'model_index.json' configuration file
# config = StableDiffusionXLPipeline.load_config("RunDiffusion/Juggernaut-XL-v9")

# import json
# print(json.dumps(dict(config), indent=2))

import torch
from diffusers import StableDiffusionXLPipeline

# 1. Load the model using the standard fp16 variant
pipe = StableDiffusionXLPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-XL-v9",
    torch_dtype=torch.float16,   # CRITICAL: Cuts VRAM usage in half
    variant="fp16",              # Downloads the lighter weight files
    use_safetensors=True
)

# 2. Maximize attention memory efficiency (Massive speed boost for 8GB cards)
# pipe.enable_xformers_memory_efficient_attention()

# 3. Aggressive step-by-step memory management for your 12GB system RAM
pipe.enable_sequential_cpu_offload()

# prompt="cinematic still of a futuristic cyberpunk city street at night, neon signs reflecting in puddles on the pavement, a lone figure in a trench coat walks away from the camera, blade runner style, volumetric lighting, ultra-detailed."
prompt="An ink sketch style illustration of a small hedgehog holding a piece of watermelon with its tiny paws, taking little bites with its eyes closed in delight."

# 4. Generate (SDXL native resolution is 1024x1024)
image = pipe(
    prompt=prompt,
    num_inference_steps=30,      # SDXL needs around 30-40 steps, unlike FLUX's 4 steps
    guidance_scale=5.0,          # Recommended CFG for Juggernaut v9 is between 3.0 and 7.0
    height=1024,
    width=1024,
).images[0]

# image.save(f"cyberpunk_city.png")
image.save(f"hedgehog.png")