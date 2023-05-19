from pathlib import Path
import argparse

import torch
from PIL import Image
from diffusers import StableUnCLIPImg2ImgPipeline



def get_latent(image, pipe, device, dtype):
    image = pipe.feature_extractor(images=image, return_tensors="pt").pixel_values
    image = image.to(device=device, dtype=dtype)
    image_embeds = pipe.image_encoder(image).image_embeds
    return image_embeds

def slerp(base, style, s):
    base_norm = base / torch.norm(base, dim=1)
    style_norm = style / torch.norm(style, dim=1)
    omega = torch.acos(torch.clamp(base_norm @ style_norm.T, -1, 1))
    res = (torch.sin((1.0 - s) * omega) / torch.sin(omega)) * base + \
        (torch.sin(s * omega) / torch.sin(omega)) * style
    return res

@torch.no_grad()
def main(base_path: str, style_path: str, influence, 
         seed: int, noise_level: int, num_inference_steps: int, guidance_scale: float):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    dtype = next(pipe.image_encoder.parameters()).dtype
    g = torch.Generator(device)
    g.manual_seed(seed)

    base_image = Image.open(base_path).convert("RGB")
    style_image = Image.open(style_path).convert("RGB")
    base_latent = get_latent(base_image, pipe, device, dtype)
    style_latent = get_latent(style_image, pipe, device, dtype)
    latent = slerp(base_latent, style_latent, influence)

    images = pipe(image_embeds=latent, 
                       noise_level=noise_level,
                       num_inference_steps=num_inference_steps,
                       guidance_scale=guidance_scale,
                       generator=g,
                       ).images
    
    base_name = Path(base_path).stem
    style_name = Path(style_path).stem
    images[0].save(f"./{base_name}_{style_name}.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('base_path', type=str, help='Base image')
    parser.add_argument('style_path', type=str, help='Style image')
    parser.add_argument(
        '--influence',
        type=float,
        default=0.4,
        help='influence of style image, between 0 and 1 (default: 0.4)'
        )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed for torch.Generator'
        )
    parser.add_argument(
        '--noise_level',
        type=int,
        default=0,
        help=' The amount of noise to add to the image embeddings. A higher noise_level increases the variance in the final un-noised images. '
        )
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=50,
        help='The number of denoising steps.'
        )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=10.,
        help='Guidance scale as defined in Classifier-Free Diffusion Guidance'
        )
    args = vars(parser.parse_args())
    main(**args)