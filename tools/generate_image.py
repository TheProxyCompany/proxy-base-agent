import dotenv
import fal_client

from agent.agent import Agent, AgentStatus
from agent.event import Event, EventState


def generate_image(self: Agent, prompt: str) -> Event:
    """
    Visualize a scene or object by prompting an image generation model to generate an image.

    Guidelines for crafting image prompts:
    - Structure prompts with a clear subject, specific details, and contextual actions.
    - Include artistic styles like "pixel art", "surrealism".

    Examples:
        Good:
            "An ancient oak tree in a misty forest, its gnarled branches stretching out like arms,
            leaves rustling softly in the twilight breeze, with rays of golden sunlight filtering
            through the canopy."
        Bad:
            "A tree in a forest."
    Args:
        prompt (str): Detailed prompt of the scene or object to visualize, including vivid visual details.
    """
    dotenv.load_dotenv()

    self.status = AgentStatus.PROCESSING

    image_url: str | None = None
    image_args = {
        "prompt": prompt + " pixelated, low resolution, 8-bit, 16x16, retro style.",
        "has_nsfw_concepts": True,
        "image_size": "square",
        "seed": self.state.seed,
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
    self.status = AgentStatus.SUCCESS

    return Event(
        state=EventState.TOOL,
        content=prompt,
        name=self.state.name + "'s image",
        image_url=image_url,
    )
