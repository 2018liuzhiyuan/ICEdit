from torch_npu.contrib import transfer_to_npu

# Use the modified diffusers & peft library
import sys
import os

# workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../icedit"))

# if workspace_dir not in sys.path:
#     sys.path.insert(0, workspace_dir)
from diffusers import FluxFillPipeline

# Below is the original library
import torch
from PIL import Image
import numpy as np
import argparse
import random
    
parser = argparse.ArgumentParser() 
parser.add_argument("--image", type=str, help="Name of the image to be edited", required=True)
parser.add_argument("--instruction", type=str, help="Instruction for editing the image", required=True)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the output image")
parser.add_argument("--flux-path", type=str, default='black-forest-labs/flux.1-fill-dev', help="Path to the model")
parser.add_argument("--lora-path", type=str, default='RiverZ/normal-lora', help="Path to the LoRA weights")
parser.add_argument("--enable-model-cpu-offload", action="store_true", help="Enable CPU offloading for the model")
parser.add_argument("--save-name", type=str, default="", help="Name of the output image")


args = parser.parse_args()
pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(args.lora_path)

if args.enable_model_cpu_offload:
    pipe.enable_model_cpu_offload(None, "npu")
else:
    pipe = pipe.to("npu")


image = Image.open(args.image)
image = image.convert("RGB")

if image.size[0] != 512:
    print("\033[93m[WARNING] We can only deal with the case where the image's width is 512.\033[0m")
    new_width = 512
    scale = new_width / image.size[0]
    new_height = int(image.size[1] * scale)
    new_height = (new_height // 8) * 8  
    image = image.resize((new_width, new_height))
    print(f"\033[93m[WARNING] Resizing the image to {new_width} x {new_height}\033[0m")

instruction = args.instruction

print(f"Instruction: {instruction}")
instruction = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}'

width, height = image.size
combined_image = Image.new("RGB", (width * 2, height))
combined_image.paste(image, (0, 0))
combined_image.paste(image, (width, 0))
mask_array = np.zeros((height, width * 2), dtype=np.uint8)
mask_array[:, width:] = 255 
mask = Image.fromarray(mask_array)

result_image = pipe(
    prompt=instruction,
    image=combined_image,
    mask_image=mask,
    height=height,
    width=width * 2,
    guidance_scale=50,
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(args.seed) if args.seed is not None else None,
).images[0]

result_image = result_image.crop((width,0,width*2,height))

os.makedirs(args.output_dir, exist_ok=True)

image_name = args.image.split("/")[-1]
if args.save_name:
    image_name = args.save_name
result_image.save(os.path.join(args.output_dir, f"{image_name}"))
print(f"\033[92mResult saved as {os.path.abspath(os.path.join(args.output_dir, image_name))}\033[0m")
