import torch
from diffusers import StableDiffusionPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline cache
_pipeline = None

def get_pipeline():
    """
    Lazy loads the Stable Diffusion pipeline.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline
        
    model_id = "runwayml/stable-diffusion-v1-5"
    
    logger.info(f"Loading Stable Diffusion model: {model_id}...")
    
    try:
        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS (Apple Silicon) acceleration.")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA acceleration.")
        else:
            device = "cpu"
            logger.warning("Using CPU. Generation will be slow.")

        # Load pipeline
        # Use float16 for MPS/CUDA to save memory and speed up
        if device != "cpu":
             pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)

        pipe = pipe.to(device)
        
        # Disable safety checker to prevent black images on false positives
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        
        # Access safety checker if needed, but for "pencil art" usually fine.
        # Minimal optimization
        if device == "mps":
             pipe.enable_attention_slicing()

        _pipeline = pipe
        logger.info("Stable Diffusion pipeline loaded successfully.")
        return _pipeline
        
    except Exception as e:
        logger.error(f"Failed to load Stable Diffusion pipeline: {e}")
        raise e

def generate_panel_sd(prompt, negative_prompt="", width=512, height=512):
    """
    Generates a comic panel using Stable Diffusion.
    """
    pipe = get_pipeline()
    
    # Generate
    try:
        image = pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            width=width, 
            height=height,
            num_inference_steps=30, # Balance between quality and speed
            guidance_scale=7.5
        ).images[0]
        
        return image
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None
