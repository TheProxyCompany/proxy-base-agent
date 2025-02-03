import logging
from collections.abc import Callable, Iterator
from typing import Any

import mlx.core as mx
from mlx_proxy.generate_step import generate_step
from mlx_proxy.samplers import make_sampler
from mlx_proxy.utils import load_model, set_max_reccomended_device_limit
from pse.structure.engine import StructuringEngine

from agent.llm.frontend import Frontend
from agent.llm.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class MLXInference(Frontend):
    """
    Front-end for MLX models.
    """

    def __init__(self, model_path: str):
        """
        Initialize the MLXFrontEnd.

        Args:
            model_path (str): The path to the model.
        """
        set_max_reccomended_device_limit()
        self.model, self.model_type = load_model(model_path)
        self.tokenizer = Tokenizer.load(model_path, self.model_type)
        self.engine = StructuringEngine(self.tokenizer._tokenizer)

    def inference(
        self,
        prompt: list[int],
        structured: bool = True,
        **kwargs,
    ) -> Iterator[int]:
        """
        A generator producing token ids based on the given prompt from the model.

        Args:
            prompt (list[int]): The input prompt.
            simple_sampler (bool): Whether to use simple sampling.
            **kwargs: Keyword arguments for the sampler.
        """
        if seed := kwargs.get("seed", None):
            mx.random.seed(seed)

        for generated_tokens, _ in generate_step(
            prompt=mx.array(prompt),
            model=self.model,
            logits_processors=[self.engine] if structured else None,
            sampler=self.make_sampler(structured, **kwargs),
            max_tokens=kwargs.get("max_tokens", 1000),
        ):
            assert isinstance(generated_tokens, mx.array)
            assert generated_tokens.ndim == 1
            tokens = generated_tokens.tolist()
            assert isinstance(tokens, list)
            for token_id in tokens:
                if token_id in self.tokenizer.stop_tokens:
                    return
                yield token_id

            if self.engine.has_reached_accept_state:
                break

    def make_sampler(self, structured: bool, **kwargs) -> Callable[..., Any]:
        """
        Return a sampler function.
        If structured is True, use the structured sampler.
        Otherwise, use the simple sampler.
        """
        temp = float(kwargs.get("temp", 1.0))
        min_p = float(kwargs.get("min_p", 0.0))
        min_tokens_to_keep = int(kwargs.get("min_tokens_to_keep", 1))
        sampler = make_sampler(
            temp=temp, min_p=min_p, min_tokens_to_keep=min_tokens_to_keep
        )
        if structured:
            return lambda x: self.engine.sample(x, sampler)
        else:
            return sampler
