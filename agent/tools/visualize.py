import os

from mflux.config.config import Config
from mflux.flux.flux import Flux1

from agent.agent import Agent
from agent.interaction import Interaction

IMAGE_PATH = "/tmp/image.png"

def visualize(
    self: Agent,
    prompt: str,
    steps: int = 4,
    guidance: int = 8,
) -> Interaction:
    """
    Prompt an image generation model to generate an image.
    This image is displayed to the user.
    Include artistic styles like "surrealism", "impressionist", "anime", "pixel art", etc.

    Args:
        prompt (str): The prompt for the image generation model.
        steps (int, optional):
            The number of inference steps to take in the image generation process.
            Defaults to 4. Trade latency for quality (1 step = 70 seconds).
        guidance (int, optional):
            The guidance scale for the image generation process.
            Defaults to 8. Trade diversity for prompt control.
    """
    if os.path.exists(IMAGE_PATH):
        os.remove(IMAGE_PATH)

    flux = Flux1.from_alias(
        alias="schnell",
        quantize=8,
    )
    image = flux.generate_image(
        seed=self.seed,
        prompt=prompt,
        config=Config(
            num_inference_steps=steps,
            height=512,
            width=512,
            guidance=guidance,
        )
    )
    image.save(path=IMAGE_PATH)

    return Interaction(
        role=Interaction.Role.ASSISTANT,
        content=f"Generated an image:\n*{prompt}*",
        title=self.name + "'s image",
        image_url=IMAGE_PATH,
        color="yellow",
        emoji="camera",
    )
