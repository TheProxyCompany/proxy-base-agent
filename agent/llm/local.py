import logging
from collections.abc import Iterable
from typing import Any

from pse.structuring_engine import StructuringEngine

from agent.llm.frontend import Frontend
from agent.system.interaction import Interaction

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
        self.engine = StructuringEngine(
            self.front_end.tokenizer._tokenizer,
            whitelist_control_tokens=self.front_end.tokenizer.whitelist_control_tokens,
            multi_token_sampling=True,
        )

    def run_inference(
        self,
        prompt: str | list[dict[str, Any]] | list[Interaction],
        **inference_kwargs,
    ) -> Iterable[int]:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (str | list[dict[str, Any]] | list[Event]): The input prompt for completion.
            **inference_kwargs: Additional keyword arguments to use for inference.
        """
        tokenizer_config = {
            "prompt": prompt,
            **inference_kwargs,
            **self.front_end.tokenizer.control_tokens.model_dump(),
        }
        encoded_prompt = self.front_end.tokenizer.encode(**tokenizer_config)
        logger.info(f"PROMPT:\n{self.front_end.tokenizer.decode(encoded_prompt)}")
        yield from self.front_end.inference(
            encoded_prompt, self.engine, **inference_kwargs
        )
