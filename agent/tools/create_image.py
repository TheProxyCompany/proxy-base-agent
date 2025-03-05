import os
import time

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.flux.flux import Flux1

from agent.agent import Agent
from agent.system.interaction import Interaction

IMAGE_PATH = "/tmp/image.png"

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

    tic = time.time()
    model_config = ModelConfig.schnell()

    flux = Flux1(
        model_config=model_config,
        quantize=8,
    )
    image = flux.generate_image(
        seed=self.seed,
        prompt=prompt + " pixelated, low resolution, 8-bit, 16x16, retro style.",
        config=Config(
            num_inference_steps=steps,
            height=1024,
            width=1024,
            guidance=guidance,
        ),
    )
    if os.path.exists(IMAGE_PATH):
        os.remove(IMAGE_PATH)

    image.save(path=IMAGE_PATH)
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
