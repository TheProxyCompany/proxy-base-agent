import logging
import time
from collections.abc import Iterable
from typing import Any

from pse.types.json import JSONSchemaSource

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
        self.engine = self.front_end.engine

    def __call__(self, *args, **kwargs) -> Iterable[int]:
        return self.run_inference(*args, **kwargs)

    def run_inference(
        self,
        prompt: str | list[dict[str, Any]] | list[Interaction],
        structure: JSONSchemaSource | None = None,
        buffer_length: int = -1,
        include_python: bool = True,
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
        tic = time.perf_counter()
        if structure:
            delimiters = self.front_end.tokenizer.control_tokens.tool_use_delimiters()
            self.engine.configure(
                structure,
                json_delimiters=delimiters,
                min_buffer_length=buffer_length,
                include_python=include_python,
            )
            toc = time.perf_counter()
            setup_time = toc - tic
            logger.debug(f"Structuring Engine configured in {setup_time:.4f}s")

        tokenizer_config = {
            "prompt": prompt,
            **inference_kwargs,
            **self.front_end.tokenizer.control_tokens.model_dump(),
        }
        encoded_prompt = self.front_end.tokenizer.encode(**tokenizer_config)
        logger.info(f"PROMPT:\n{self.front_end.tokenizer.decode(encoded_prompt)}")
        for token_id in self.front_end(encoded_prompt, **inference_kwargs):
            logger.debug(f"generated token: {token_id}")
            encoded_prompt.append(token_id)
            yield token_id

        toc = time.perf_counter()
        generation_time = toc - tic
        logger.debug(f"Generation time: {generation_time:.4f}s")

    def __getattribute__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            front_end = object.__getattribute__(self, "front_end")
            if name in front_end.__dict__:
                return front_end.__dict__[name]
            return None
