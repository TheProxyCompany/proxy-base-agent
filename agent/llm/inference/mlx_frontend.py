import glob
import importlib
import logging
import time
from collections.abc import Iterator
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import categorical_sampling, min_p_sampling
from mlx_lm.utils import get_model_path, load_config
from pse.structure.engine import StructuringEngine

from agent.llm.inference.frontend import Frontend
from agent.llm.util.kv_cache import KeyValueCache

# from agent.llm.util.reusable_cache import ReusableKVCache
from agent.llm.util.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class MLXFrontEnd(Frontend):
    """
    Front-end for MLX models.
    """

    def __init__(self, model_path: str):
        """
        Initialize the MLXFrontEnd.

        Args:
            model_path (str): The path to the model.
        """
        self.model, self.model_type = self.load_model(model_path)
        # self.cache = ReusableKVCache.from_model(self.model)
        self.cache = KeyValueCache.from_model(self.model)
        self.tokenizer = Tokenizer.load(model_path, self.model_type)
        self.engine = StructuringEngine(self.tokenizer._tokenizer)
        self.computed_prompt_tokens = []

    def inference(self, prompt: list[int], **kwargs) -> Iterator[Frontend.Output]:
        """
        A generator producing token ids based on the given prompt from the model.

        Args:
            prompt (list[int]): The input prompt.
            **kwargs: Keyword arguments for the sampler.
        """
        if seed := kwargs.get("seed", None):
            mx.random.seed(seed)

        mlx_prompt = mx.array(prompt)
        # if isinstance(self.cache[0], ReusableKVCache) and self.computed_prompt_tokens:
        #     tic = time.perf_counter()
        #     i = 0
        #     for i, t in enumerate(self.computed_prompt_tokens):
        #         if i >= len(mlx_prompt) - 1 or mlx_prompt[i] != t:
        #             break
        #     for layer_cache in self.cache:
        #         assert isinstance(layer_cache, ReusableKVCache)
        #         layer_cache.reuse(len(mlx_prompt), i)
        #     logger.debug(f"Reusing KVCache for {i}/{len(mlx_prompt)} tokens")
        #     y = mlx_prompt[i:]
        #     mx.metal.clear_cache()
        #     toc = time.perf_counter()
        #     reuse_time = toc - tic
        #     logger.debug(f"Reuse time: {reuse_time:.4f}s")
        # else:
        y = mlx_prompt

        if not self.computed_prompt_tokens:
            self.computed_prompt_tokens = prompt

        model_output = self.inference_step(y, **kwargs)
        y, logprobs = model_output.tokens, model_output.logprobs
        mx.async_eval(model_output.tokens, logprobs)
        mx.eval(y)

        while True:
            yield model_output
            new_model_output = self.inference_step(y, **kwargs)
            new_y, new_logprobs = new_model_output.tokens, new_model_output.logprobs
            mx.async_eval(new_y, new_logprobs)

            model_output = new_model_output
            y, logprobs = new_y, new_logprobs

    def inference_step(self, prompt: mx.array, **sampler_kwargs) -> Frontend.Output:
        """
        A single step of inference on the given prompt from the model.

        Args:
            prompt (mx.array): The input prompt.
            **sampler_kwargs: Keyword arguments for the sampler.
        returns:
            Result: The result of the generation step.
        """
        tic = time.perf_counter()
        logits = self.model(prompt[None], cache=self.cache)
        logits = logits[:, -1, :]
        assert isinstance(logits, mx.array)
        toc = time.perf_counter()
        inference_time = toc - tic
        logger.debug(f"Model inference time: {inference_time:.4f}s")

        engine_time = 0.0
        if self.engine:
            tic = time.perf_counter()
            logits = self.engine(logits[0, :])
            toc = time.perf_counter()
            engine_time = toc - tic
            logger.debug(f"Engine time: {engine_time:.4f}s")

        tic = time.perf_counter()
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        token_ids = (
            self.engine.sample(logprobs, self.sample_tokens, **sampler_kwargs)
            if self.engine
            else self.sample_tokens(logprobs, **sampler_kwargs).tolist()
        )
        assert isinstance(token_ids, list)
        self.computed_prompt_tokens.extend(token_ids)
        toc = time.perf_counter()
        sampling_time = toc - tic
        logger.debug(f"Sampling time: {sampling_time:.4f}s")

        return Frontend.Output(
            mx.array(token_ids, dtype=prompt.dtype),
            token_ids,
            logprobs,
            self.engine is not None and self.engine.is_within_value,
            inference_time,
            engine_time,
            sampling_time,
        )

    @staticmethod
    def sample_tokens(logprobs: mx.array, **kwargs) -> mx.array:
        """
        Sample tokens from the given logprobs.
        This function is used by the structuring engine.
        Easily extendable.

        Args:
            logprobs (mx.array): The logprobs to sample from.
            **kwargs: Keyword arguments for the sampler.
        Returns:
            mx.array: The sampled tokens.
        """
        temp = float(kwargs.get("temperature", 1.0))
        min_p = float(kwargs.get("min_p", 0.0))
        min_tokens_to_keep = int(kwargs.get("min_tokens_to_keep", 1))

        token: mx.array
        if min_p > 0.0:
            token = min_p_sampling(logprobs[None], min_p, min_tokens_to_keep, temp)
        elif temp > 0.0:
            token = categorical_sampling(logprobs[None], temp)
        else:
            token = mx.argmax(logprobs[None])

        return token

    @staticmethod
    def load_model(model_path: str) -> tuple[nn.Module, str]:
        """
        Load and initialize the model from a given path.

        Args:
            model_path (Path): The path to load the model from.
        Returns:
            nn.Module: The loaded and initialized model.
        """
        path = get_model_path(model_path)
        config = load_config(path)
        weight_files = glob.glob(str(path / "model*.safetensors"))
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        model_class, model_args_class = MLXFrontEnd.get_model_architecture(config)
        model_args = model_args_class.from_dict(config)
        model = model_class(model_args)
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)

        if (quantization := config.get("quantization", None)) is not None:
            nn.quantize(model, **quantization)

        model.load_weights(list(weights.items()))
        assert isinstance(model, nn.Module)
        mx.eval(model.parameters())
        model.eval()
        return model, config.get("model_type", "chatml")

    @staticmethod
    def get_model_architecture(config: dict[str, Any]):
        """
        Retrieve the model and model args classes based on the configuration.

        Args:
            config (dict): The model configuration.

        Returns:
            A tuple containing the Model class and the ModelArgs class.
        """
        model_type = config["model_type"]
        model_type = {
            "mistral": "llama",
            "phi-msft": "phixtral",
            "falcon_mamba": "mamba",
        }.get(model_type, model_type)

        arch = None
        try:
            try:
                arch = importlib.import_module(f"agent.llm.models.{model_type}")
            except ImportError:
                arch = importlib.import_module(f"mlx_lm.models.{model_type}")
        except ImportError:
            msg = f"Model type {model_type} not supported."
            logging.error(msg)

        if arch is None:
            raise ValueError("No model architecture found for the given model type.")

        return arch.Model, arch.ModelArgs
