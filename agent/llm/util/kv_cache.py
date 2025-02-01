from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import mlx.core as mx
import mlx.nn as nn


class KeyValueCache:
    """
    Base cache class for handling cache states.
    """

    def __init__(self, offset: int | None = None, step_size: int | None = None):
        """
        Initialize the cache.

        Args:
            offset: The offset of the cache.
            step_size: The step size of the cache.
        """
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.metadata: dict[str, Any] = {}

        self._offset = offset or 0
        self._step_size = step_size or 256

    @classmethod
    def from_model(cls, model: nn.Module) -> list[Self]:
        """
        Factory method to create a list of KeyValueCache instances for each layer in the model.
        """
        if not hasattr(model, "layers") or not isinstance(model.layers, Sequence):
            raise ValueError("Model must have an attribute 'layers'")
        num_layers = len(model.layers)
        return [cls() for _ in range(num_layers)]

    @property
    def size(self) -> int:
        """
        Returns the size of the cache.
        """
        if self.keys is None:
            return 0

        return self.keys.shape[2]

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        """
        Retrieve the cache's state.
        """
        if self.keys is None or self.values is None:
            raise ValueError("Cache is not initialized")

        if self._offset == self.size:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self._offset, :],
                self.values[..., : self._offset, :],
            )

    @state.setter
    def state(self, new_state: tuple[mx.array, mx.array]) -> None:
        self.keys, self.values = new_state
        self._offset = self.size

    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """
        Update the cache with new keys and values and return the updated cache.

        Returns:
            tuple[mx.array, mx.array]: The updated cache.
        """
        if keys.shape[2] + self._offset > self.size:
            # The new keys and values exceed the cache size,
            # so we need to extend the cache.
            batch_size = keys.shape[0]
            num_attention_heads = keys.shape[1]
            num_keys = keys.shape[2]
            embedding_dimension = keys.shape[3]

            # Calculate the number of chunks needed to store the new keys and values.
            num_chunks = (self._step_size + num_keys - 1) // self._step_size
            kv_shape = (
                batch_size,
                num_attention_heads,
                num_chunks * self._step_size,
                embedding_dimension,
            )
            new_cache_space = mx.zeros(kv_shape, keys.dtype)
            # Extend the cache with the extra allocation.
            if self.keys is None or self.values is None:
                self.keys = new_cache_space
                self.values = new_cache_space
            else:
                if self._offset % self._step_size != 0:
                    self.keys = self.keys[..., : self._offset, :]
                    self.values = self.values[..., : self._offset, :]
                self.keys = mx.concatenate([self.keys, new_cache_space], axis=2)
                self.values = mx.concatenate([self.values, new_cache_space], axis=2)

        assert self.keys is not None and self.values is not None
        self.keys[..., self._offset : self._offset + keys.shape[2], :] = keys
        self.values[..., self._offset : self._offset + values.shape[2], :] = values
        self._offset += keys.shape[2]
        return self.keys[..., : self._offset, :], self.values[..., : self._offset, :]
