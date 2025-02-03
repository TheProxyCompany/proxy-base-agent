import dotenv
import fal_client

from agent.agent import Agent
from agent.interaction import Interaction


def visualize(self: Agent, prompt: str) -> Interaction:
    """
    Prompt an image generation model to generate an image.
    This image is displayed to the user.

    Guidelines for crafting image prompts:
    - Structure prompts with a clear subject, specific details, and contextual actions.
    - Include artistic styles like "surrealism", "impressionist", "anime", "pixel art", etc.

    Args:
        prompt (str): The prompt for the image generation model.
    """
    dotenv.load_dotenv()

    image_url: str | None = None
    image_args = {
        "prompt": prompt + " pixelated, low resolution, 8-bit, 16x16, retro style.",
        "has_nsfw_concepts": True,
        "image_size": "square",
        "seed": self.seed,
        "guidance_scale": 11.0,
    }
    handler = fal_client.submit("fal-ai/flux/dev", image_args)
    visualization = handler.get()

    if (
        not visualization
        or "images" not in visualization
        or not visualization["images"]
    ):
        raise ValueError("No valid images in visualization response")

    image_url = visualization["images"][0].get("url")

    return Interaction(
        role=Interaction.Role.ASSISTANT,
        content=f"*{prompt}*",
        title=self.name + "'s image",
        image_url=image_url,
        color="yellow",
        emoji="camera",
    )
