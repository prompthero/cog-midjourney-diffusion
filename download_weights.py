import os
import torch
from diffusers import StableDiffusionPipeline


model_id = "prompthero/midjourney-v4-diffusion"
cache_dir = "midjourney-diffusion-cache"
os.makedirs(cache_dir, exist_ok=True)


pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    use_auth_token=sys.argv[1],
)
