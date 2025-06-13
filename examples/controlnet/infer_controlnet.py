from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import time

base_model_path = "runwayml/stable-diffusion-v1-5"
controlnet_path = "/root/shreyas/output/checkpoint-100/controlnet"

print("Loading models...")
start_time = time.time()

# Load ControlNet
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

# Load pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,  # Disable safety checker for faster loading
    requires_safety_checker=False,
)

# Move to GPU
pipe = pipe.to("cuda")

# Speed up diffusion process
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

# Optional: Enable model CPU offloading to save VRAM
# pipe.enable_model_cpu_offload()

print(f"Models loaded in {time.time() - start_time:.2f} seconds")

# Load control image
control_image = load_image("./U4Xfpgt8cBxyUdE69hxcP6fyoofziLSSpuEduRlPW0kpPMaeaUi2fY8qkwkTiyg620240920145329.png")
prompt = "Generate a realistic drop-shadow for the given product."

# First inference (includes compilation time)
print("Starting first inference (includes compilation)...")
start_time = time.time()
generator = torch.manual_seed(0)
image = pipe(
    prompt, 
    num_inference_steps=20, 
    generator=generator, 
    image=control_image
).images[0]
print(f"First inference completed in {time.time() - start_time:.2f} seconds")

image.save("./output.png")

# Subsequent inferences will be much faster
print("Running second inference...")
start_time = time.time()
generator = torch.manual_seed(1)
image2 = pipe(
    prompt, 
    num_inference_steps=20, 
    generator=generator, 
    image=control_image
).images[0]
print(f"Second inference completed in {time.time() - start_time:.2f} seconds")