
from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

# Load the model once
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32 )
pipe = pipe.to("cpu")  # Change to "mps" if on Apple GPU

# Path for cached images
CACHE_DIR = Path("cache_images")
CACHE_DIR.mkdir(exist_ok=True)

def generate_image(prompt: str, width: int = 512, height: int = 512,
                   num_inference_steps: int = 50, guidance_scale: float = 7.5):
    """
    Generates an image from the given prompt.
    If the image already exists in cache, returns the cached image path.

    Parameters:
    - prompt: Text prompt in English
    - width: Image width
    - height: Image height
    - num_inference_steps: Number of denoising steps
    - guidance_scale: Classifier-free guidance scale
    """
    # Safe filename based on prompt and options
    safe_name = "_".join(prompt.lower().split()) + f"_{width}x{height}_{num_inference_steps}_{guidance_scale}.png"
    img_path = CACHE_DIR / safe_name

    # Return cached image if it exists
    if img_path.exists():
        return img_path

    # Generate the image
    image = pipe(prompt, width=width, height=height,
                 num_inference_steps=num_inference_steps,
                 guidance_scale=guidance_scale).images[0]
    image.save(img_path)
    return img_path
