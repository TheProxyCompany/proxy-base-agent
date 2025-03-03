import logging
from collections.abc import Iterable
from typing import Any

from agent.interaction import Interaction
from agent.llm.frontend import Frontend

logger = logging.getLogger(__name__)


class LocalInference:

    def __init__(self, model_path: str, frontend: str | None = "mlx"):
        """
        Initialize the Inference class.

        Args:
            model_path (str): Path to the model.

        This method sets up the necessary components for inference, including:
        - Loading the model configuration
        - Initializing the tokenizer and model
        - Setting up caches and data structures for efficient inference
        """
        self.front_end = Frontend.from_path(model_path, frontend)

    def run_inference(
        self,
        prompt: str | list[dict[str, Any]] | list[Interaction],
        **inference_kwargs,
    ) -> Iterable[int]:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (str | list[dict[str, Any]] | list[Event]): The input prompt for completion.
            structure (StructuringEngine.StructureType | None): schema to constrain the output.
            buffer_length (int): The length of the buffer to use for inference.
            max_tokens (int): The maximum number of tokens to generate.
            **inference_kwargs: Additional keyword arguments to use for inference.

        Returns:
            Iterable[ModelOutput]: The output from the model, each element are sampled tokens & logprobs
        """
        tokenizer_config = {
            "prompt": prompt,
            **inference_kwargs,
            **self.front_end.tokenizer.control_tokens.model_dump(),
        }
        encoded_prompt = self.front_end.tokenizer.encode(**tokenizer_config)
        logger.info(f"PROMPT:\n{self.front_end.tokenizer.decode(encoded_prompt)}")
        for token_id in self.front_end(encoded_prompt, **inference_kwargs):
            encoded_prompt.append(token_id)
            yield token_id
