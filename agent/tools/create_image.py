"""
Image generation tool for the Proxy Base Agent.

This module provides a tool for generating images from text prompts using
the Flux diffusion model. It creates stylized pixel art visualizations that
can be displayed in the agent interface.
"""

import os
import time
import tempfile
import uuid

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.flux.flux import Flux1

from agent.agent import Agent
from agent.system.interaction import Interaction

# Path for storing generated images
# Using /tmp for temporary storage - in production, consider a more robust solution
IMAGE_PATH = os.path.join(tempfile.gettempdir(), f"agent_image_{uuid.uuid4()}.png")

def create_image(
    self: Agent,
    prompt: str,
    steps: int = 4,
    guidance: int = 8,
) -> Interaction:
    """
    Generate and display an image based on a text prompt.

    This tool has a permanent pixel art style (16x16 resolution, 8-bit aesthetic)
    to create quick, stylized visualizations. The generated image is saved to a
    temporary file and displayed to the user.

    The prompt should reflect your (the agent's) visual interpretation of the scene
    or concept. Consider incorporating artistic elements such as:
    - Artistic movements (surrealism, impressionism, art nouveau)
    - Visual styles (anime, pixel art, watercolor, oil painting)
    - Compositional elements (perspective, lighting, color palette)

    Arguments:
        prompt: Descriptive text that will be used to generate the image.
            The prompt will automatically be enhanced with pixel art modifiers.
        steps: Number of diffusion steps for image generation.
            Each step takes ~30 seconds. More steps = higher quality.
            Default: 4 steps (≈ 120 seconds)
        guidance: Controls how closely the image follows the prompt.
            Higher values (>7) = more literal interpretation but less creative variety
            Lower values (<7) = more creative but may diverge from prompt
            Default: 8.0
    """
    if not prompt:
        raise ValueError("Prompt is required, cannot visualize an empty prompt.")

    # Start timing the image generation process
    tic = time.time()
    
    # Initialize model with Schnell config (optimized for speed)
    model_config = ModelConfig.schnell()

    # Create Flux model instance with 8-bit quantization for efficiency
    flux = Flux1(
        model_config=model_config,
        quantize=8,  # Use 8-bit quantization to reduce memory usage
    )
    
    # Generate the image, appending pixel art style modifiers to the prompt
    # This ensures consistent visual style across all generated images
    image = flux.generate_image(
        seed=self.seed,  # Use the agent's seed for reproducibility
        prompt=prompt + " pixelated, low resolution, 8-bit, 16x16, retro style.",
        config=Config(
            num_inference_steps=steps,  # Controls generation quality vs. speed
            height=1024,                # Output resolution
            width=1024,                 # Output resolution
            guidance=guidance,          # Controls adherence to prompt vs. creativity
        ),
    )
    
    # Clean up any existing image at the target path to avoid permission issues
    if os.path.exists(IMAGE_PATH):
        os.remove(IMAGE_PATH)

    # Save the generated image to the temporary file location
    image.save(path=IMAGE_PATH)
    
    # Calculate and format the total generation time
    toc = time.time()
    total_time = f"{toc - tic:.2f} seconds"
    result = f"Generated an image based on the prompt: {prompt} in {total_time}. The image was displayed to the user."
    return Interaction(
        role=Interaction.Role.TOOL,
        content=result,
        title=self.name + "'s image",
        image_url=IMAGE_PATH,
        color="bright_yellow",
        emoji="camera",
    )
