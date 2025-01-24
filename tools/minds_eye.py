import dotenv
import fal_client

from agent.agent import Agent
from agent.message import Message, MessageState


def minds_eye(self: Agent, inner_thoughts: str, scene_description: str) -> Message:
    """
    Visualize a scene or object by generating a detailed mental image to enhance creativity and imagination.

    Guidelines for crafting image prompts:
    - Use vivid, sensory-rich descriptions to authentically depict scenes or objects.
    - Structure prompts with a clear subject, specific details, and contextual actions.
    - Incorporate inner_thoughts to add depth and nuance.
    - Balance detail with conciseness for clarity.
    - Include artistic styles like "impressionist", "pixel art", "surrealism", or "photorealism".

    Examples:
        Good:
            "An ancient oak tree in a misty forest, its gnarled branches stretching out like arms,
            leaves rustling softly in the twilight breeze, with rays of golden sunlight filtering
            through the canopy."
        Suboptimal:
            "A tree in a forest."
    Args:
        inner_thoughts (str): Reflective thoughts leading up to the visualization.
        scene_description (str): Detailed description of the scene or object to visualize, including vivid visual details.

    Returns:
        Message: Contains the visualization result, including the image URL if successful.
    """
    dotenv.load_dotenv()

    try:
        handler = fal_client.submit(
            "fal-ai/flux/dev",
            arguments={
                "prompt": scene_description + " pixelated, low resolution, 8-bit, 16x16, retro style.",
                "has_nsfw_concepts": True,
                "image_size": "square",
                "seed": self.state.seed,
                "guidance_scale": 11.0
            },
        )
        visualization = handler.get()

        if not visualization or "images" not in visualization or not visualization["images"]:
            raise ValueError("No valid images in visualization response")

        image_url = visualization["images"][0].get("url")
        if not image_url:
            raise ValueError("No URL found in the first image of visualization response")

        return Message(
            role="ipython",
            content=scene_description,
            state=MessageState.TOOL_RESULT,
            name=self.state.name + "'s imagination",
            inner_thoughts=inner_thoughts,
            feelings=scene_description,
            image_path=image_url
        )
    except Exception as e:
        error_message = f"Failed to generate visualization: {e}"
        return Message(
            role="ipython",
            content=error_message,
            state=MessageState.TOOL_ERROR,
            name=self.state.name + "'s imagination",
            inner_thoughts=inner_thoughts
        )
