import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

import mlx.core as mx
from mlx_proxy.cache import BaseCache
from mlx_proxy.generate_step import generate_step
from mlx_proxy.samplers import make_sampler
from mlx_proxy.utils import load_model, set_max_reccomended_device_limit
from pse.structuring_engine import StructuringEngine

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
        self.model, _ = load_model(model_path)
        self.tokenizer = Tokenizer.load(model_path)
        self.cache: list[BaseCache] = []
        self.processed_token_ids = []

    def inference(
        self,
        prompt: list[int],
        engine: StructuringEngine,
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

        if not self.cache:
            self.cache = BaseCache.make_kv_cache(
                self.model,
                max_kv_size=kwargs.get("max_kv_size", None),
                reusable=kwargs.get("reuse_prompt_cache", False),
            )

        for generated_tokens, _ in generate_step(
            prompt=prompt,
            model=self.model,
            prompt_cache=self.cache,
            logits_processors=[engine.process_logits],
            sampler=self.make_sampler(engine, **kwargs),
            max_tokens=kwargs.get("max_tokens", 1000),
            reuse_prompt_cache=kwargs.get("reuse_prompt_cache", False),
            computed_ids=self.processed_token_ids,
        ):
            assert isinstance(generated_tokens, mx.array)
            assert generated_tokens.ndim == 1
            tokens = generated_tokens.tolist()
            assert isinstance(tokens, list)
            for token_id in tokens:
                if token_id in self.tokenizer.stop_tokens:
                    break
                yield token_id

            if engine.has_reached_accept_state:
                break

    def make_sampler(self, engine: StructuringEngine, **kwargs) -> Callable[..., Any]:
        """
        Return a sampler function.
        If structured is True, use the structured sampler.
        Otherwise, use the simple sampler.
        """
        temp = float(kwargs.get("temp", 1.0))
        min_p = float(kwargs.get("min_p", 0.02))
        min_tokens_to_keep = int(kwargs.get("min_tokens_to_keep", 1))
        sampler = make_sampler(
            temp=temp,
            min_p=min_p,
            min_tokens_to_keep=min_tokens_to_keep
        )
        return lambda x: engine.sample(x, sampler)

    def supports_reusing_prompt_cache(self) -> bool:
        return True

    def save_cache_to_file(self, file_path: str, computed_ids: list[int]) -> None:
        metadata = {"computed_ids": json.dumps(computed_ids)}
        BaseCache.save_cache(file_path, self.cache, metadata)

    def load_cache_from_file(self, file_path: str) -> tuple[list[BaseCache], list[int]]:
        cached = BaseCache.load_cache(file_path, return_metadata=True)
        if isinstance(cached, tuple):
            metadata = cached[1]
            computed_ids_encoded_str = metadata["computed_ids"]
            computed_ids = json.loads(computed_ids_encoded_str)
            assert isinstance(computed_ids, list)
            return cached[0], computed_ids
        else:
            return cached, []
