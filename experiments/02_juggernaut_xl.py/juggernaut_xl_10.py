import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-X-v10", 
    torch_dtype=torch.float16,
    # device_map="auto"
)

pipe.enable_sequential_cpu_offload()

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
prompt = "high-resolution portrait of a Renaissance noblewoman, intricate lace collar, holding an ancient book, dominant colors deep red and gold, style reminiscent of Vermeer's lighting techniques, mood thoughtful, lighting soft, natural window light, perspective close-up, texture rich fabrics and aged paper, cultural elements European Renaissance elegance"
# prompt = "A cliffside cultural retreat using rammed basalt composite, storm-rated laminated fins, and elevated walkways hugging rock edges. Interiors include dark polished schist floors, recessed ember lights, and echo-minimizing wool baffles. Shot in golden-hour edge illumination, diagonal cliffside angle."
# prompt = "A mangrove research pier with bio-friendly screw anchors, ventilated wood-plastic composite decking, and salt-air tolerant façades. Interiors feature mosquito-screened lab bays, resin benches, and low-Kelvin night lighting. Captured in humid early-morning haze, low angle along boardwalk."

image = pipe(prompt,
    num_inference_steps=30,      # SDXL needs around 30-40 steps
    guidance_scale=5.0,          # Recommended CFG for Juggernaut v10 is between 3.0 and 7.0
    height=1024,
    width=1024
).images[0]

image.show()
image.save("juggernaut_xl_10_01.png")


# TODO: Batch Size?