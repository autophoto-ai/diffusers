#!/usr/bin/env python3
"""
Example script to use a LoRA-trained ControlNet model for inference.
This demonstrates how to load and use the LoRA adapters with the base model.
"""

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Inference with LoRA ControlNet")
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        default="runwayml/stable-diffusion-v1-5",
        help="Path to the base Stable Diffusion model"
    )
    parser.add_argument(
        "--lora_model_path", 
        type=str, 
        required=True,
        help="Path to the trained LoRA ControlNet model"
    )
    parser.add_argument(
        "--control_image", 
        type=str, 
        required=True,
        help="Path to the control conditioning image"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="a beautiful landscape",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./generated_image.png",
        help="Output path for generated image"
    )
    parser.add_argument(
        "--base_controlnet", 
        type=str, 
        default="lllyasviel/sd-controlnet-canny",
        help="Base ControlNet model to load LoRA adapters onto"
    )
    
    args = parser.parse_args()
    
    # Load the base ControlNet model
    print(f"Loading base ControlNet from: {args.base_controlnet}")
    controlnet = ControlNetModel.from_pretrained(
        args.base_controlnet,
        torch_dtype=torch.float16
    )
    
    # Create the pipeline first
    print(f"Loading base Stable Diffusion model from: {args.base_model_path}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Load LoRA adapters for ControlNet
    if os.path.exists(args.lora_model_path):
        try:
            print(f"Loading LoRA adapters from: {args.lora_model_path}")
            
            # Load LoRA weights for ControlNet
            pipe.load_lora_weights(args.lora_model_path, adapter_name="controlnet_lora")
            
            # Set the LoRA scale (you can adjust this value)
            pipe.set_adapters(["controlnet_lora"], adapter_weights=[1.0])
            
            print("LoRA adapters loaded successfully!")
            
        except Exception as e:
            print(f"Failed to load LoRA adapters: {e}")
            print("Continuing with base ControlNet model...")
    else:
        print(f"LoRA model path does not exist: {args.lora_model_path}")
        print("Using base ControlNet model...")
    
    # Enable memory efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers memory efficient attention enabled")
    except:
        print("xformers not available, using standard attention")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    print(f"Pipeline moved to device: {device}")
    
    # Load and process the control image
    print(f"Loading control image: {args.control_image}")
    control_image = Image.open(args.control_image).convert("RGB")
    
    # Generate the image
    print(f"Generating image with prompt: '{args.prompt}'")
    print("Generation parameters:")
    print(f"  - Inference steps: 20")
    print(f"  - Guidance scale: 7.5")
    print(f"  - ControlNet conditioning scale: 1.0")
    
    with torch.autocast(device):
        image = pipe(
            prompt=args.prompt,
            image=control_image,
            num_inference_steps=20,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            generator=torch.Generator(device=device).manual_seed(42)  # For reproducible results
        ).images[0]
    
    # Save the result
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
    image.save(args.output_path)
    print(f"Generated image saved to: {args.output_path}")

if __name__ == "__main__":
    main()

"""
Example usage commands:

# Basic usage with LoRA ControlNet:
python inference_controlnet_lora.py \
    --lora_model_path /root/shreyas/output_fusing_fill50k_lora/checkpoint-11500 \
    --control_image /root/shreyas/diffusers/examples/controlnet/conditioning_image_1.png \
    --prompt "red circle with blue background" \
    --output_path ./output.png

# With custom base ControlNet and model:
python inference_controlnet_lora.py \
    --base_model_path ./custom/stable-diffusion-model \
    --base_controlnet lllyasviel/sd-controlnet-depth \
    --lora_model_path ./path/to/lora/checkpoint \
    --control_image ./depth_map.png \
    --prompt "photorealistic portrait of a person" \
    --output_path ./portrait_output.png

# Minimal command (using defaults):
python inference_controlnet_lora.py \
    --lora_model_path ./my_lora_checkpoint \
    --control_image ./my_control.png

Note: Make sure your LoRA checkpoint directory contains the adapter files.
The script will automatically detect and load LoRA weights if available.
"""