import glob
import logging
import time
from collections.abc import Iterator

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.sample_utils import categorical_sampling, min_p_sampling
from mlx_lm.utils import _get_classes, get_model_path, load_config
from model_inference.chat_templates import get_control_tokens, load_chat_template
from model_inference.front_ends import FrontEnd
from model_inference.utils.reuseable_cache import ReusableKVCache
from model_inference.utils.tokenizer_wrapper import load_tokenizer
from pse.structuring_engine import StructuringEngine

logger = logging.getLogger(__name__)


class MLXFrontEnd(FrontEnd):
    """
    Front-end for MLX models.
    """

    def __init__(self, model_path: str):
        """
        Initialize the MLXFrontEnd.

        Args:
            model_path (str): The path to the model.
        """
        self.load_model(model_path)
        self.initialize_cache(self.model)
        self.control_tokens = get_control_tokens(self.model_type)
        self.tokenizer = load_tokenizer(model_path, self.control_tokens)
        self.tokenizer._tokenizer.chat_template = load_chat_template()
        self.engine = StructuringEngine(self.tokenizer._tokenizer)

    def __call__(self, prompt: list[int], **kwargs) -> Iterator[FrontEnd.Result]:
        return self.generate(prompt, **kwargs)

    def generate(
        self,
        prompt: list[int],
        temp: float = 1.0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        previous_prompt_tokens: list[int] | None = None,
    ) -> Iterator[FrontEnd.Result]:
        mlx_prompt = mx.array(prompt)
        if isinstance(self.cache[0], ReusableKVCache) and previous_prompt_tokens:
            tic = time.perf_counter()
            i = 0
            for i, t in enumerate(previous_prompt_tokens):
                if i >= len(mlx_prompt) - 1 or mlx_prompt[i] != t:
                    break
            for layer_cache in self.cache:
                assert isinstance(layer_cache, ReusableKVCache)
                layer_cache.reuse(len(mlx_prompt), i)
            logger.debug(f"Reusing KVCache for {i}/{len(mlx_prompt)} tokens")
            y = mlx_prompt[i:]
            mx.metal.clear_cache()
            toc = time.perf_counter()
            reuse_time = toc - tic
            logger.debug(f"Reuse time: {reuse_time:.4f}s")
        else:
            y = mlx_prompt

        sampler_kwargs = {
            "temperature": temp,
            "min_p": min_p,
            "min_tokens_to_keep": min_tokens_to_keep,
        }

        result = self.sample(y, **sampler_kwargs)
        y, logprobs = result.tokens, result.logprobs
        mx.async_eval(result.tokens, logprobs)
        mx.eval(y)

        while True:
            yield result
            next_result = self.sample(y, **sampler_kwargs)
            next_y, next_logprobs = next_result.tokens, next_result.logprobs
            mx.async_eval(next_y, next_logprobs)

            result = next_result
            y, logprobs = next_y, next_logprobs

    def sample(
        self,
        prompt: mx.array,
        **sampler_kwargs,
    ) -> FrontEnd.Result:
        """
        A generator producing token ids based on the given prompt from the model.

        Args:
            prompt (mx.array): The input prompt.
            model (nn.Module): The model to use for generation.
            engine (StructuringEngine): The engine to use for generation.
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

        logprobs = logits - mx.logsumexp(logits, keepdims=True)

        def __sampler(logprobs: mx.array, **kwargs) -> mx.array:
            temp = float(kwargs.get("temperature", 1.0))
            min_p = float(kwargs.get("min_p", 0.0))
            min_tokens_to_keep = int(kwargs.get("min_tokens_to_keep", 1))

            token: mx.array
            if min_p > 0.0:
                token = min_p_sampling(logprobs[None], min_p, min_tokens_to_keep, temp)
            elif temp > 0.0:
                token = categorical_sampling(logprobs, temp)
            else:
                token = mx.argmax(logprobs, axis=-1)

            return token

        tic = time.perf_counter()
        if self.engine:
            token_ids = self.engine.sample(logprobs, __sampler, **sampler_kwargs)
        else:
            token_ids = __sampler(logprobs, **sampler_kwargs).tolist()
            assert isinstance(token_ids, list)

        toc = time.perf_counter()
        sampling_time = toc - tic
        logger.debug(f"Sampling time: {sampling_time:.4f}s")

        return FrontEnd.Result(
            mx.array(token_ids, dtype=prompt.dtype),
            token_ids,
            logprobs,
            self.engine.has_reached_accept_state if self.engine else False,
            inference_time,
            engine_time,
            sampling_time,
        )

    def initialize_cache(self, model: nn.Module, max_kv_size: int | None = None) -> None:
        """
        Initialize the cache for the model.

        Args:
            model (nn.Module): The model for which to initialize the cache.
            max_kv_size (Optional[int]): The maximum size of the key-value cache.

        Returns:
            List[Cache]: A list of initialized caches for each layer of the model.
        """
        if model and hasattr(model, "make_cache"):
            self.cache = model.make_cache()  # type: ignore reportOptionalCall
        else:
            assert model.args is not None
            n_kv_heads = model.args.num_key_value_heads
            head_dim = (
                model.args.head_dim
                or model.args.hidden_size // model.args.num_attention_heads
            )
            kv_heads = (
                [n_kv_heads] * len(model.layers or [])
                if isinstance(n_kv_heads, int)
                else n_kv_heads
            )
            if max_kv_size is not None:
                self.cache = [
                    RotatingKVCache(max_kv_size, n) for n in range(len(model.layers or []))
                ]
            else:
                self.cache = [ReusableKVCache(head_dim, n) for n in kv_heads or []]

    def load_model(self, model_path: str, lazy: bool = False) -> None:
        """
        Load and initialize the model from a given path.

        Args:
            model_path (Path): The path to load the model from.
            lazy (bool): If False eval the model parameters to make sure they are
                loaded in memory before returning, otherwise they will be loaded
                when needed. Default: ``False``
            model_config (dict, optional): Configuration parameters for the model.
                Defaults to an empty dictionary.
            get_model_classes (Callable[[dict], Tuple[Type[nn.Module], Type]], optional):
                A function that returns the model class and model args class given a config.
                Defaults to the _get_classes function.

        Returns:
            nn.Module: The loaded and initialized model.

        Raises:
            FileNotFoundError: If the weight files (.safetensors) are not found.
            ValueError: If the model class or args class are not found or cannot be instantiated.
        """
        path = get_model_path(model_path)
        config = load_config(path)
        model_type: str = config.get("model_type", "chatml")

        weight_files = glob.glob(str(path / "model*.safetensors"))

        if not weight_files:
            # Try weight for back-compat
            weight_files = glob.glob(str(path / "weight*.safetensors"))

        if not weight_files:
            logging.error(f"No safetensors found in {path}")
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        model_class, model_args_class = _get_classes(config=config)

        model_args = model_args_class.from_dict(config)
        model = model_class(model_args)

        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)

        if (quantization := config.get("quantization", None)) is not None:
            # Handle legacy models which may not have everything quantized
            def class_predicate(p, m):
                if not hasattr(m, "to_quantized"):
                    return False
                return f"{p}.scales" in weights

            nn.quantize(
                model,
                **quantization,
                class_predicate=class_predicate,
            )

        model.load_weights(list(weights.items()))

        if not lazy:
            mx.eval(model.parameters())

        model.eval()
        self.model = model
        self.model_type = model_type
