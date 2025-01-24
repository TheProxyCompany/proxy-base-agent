from collections.abc import Sequence
from typing import Any

from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback

from agent.api.chat_completion import get_chat_completion
from agent.api.text_completion import get_text_completion
from config import MindConfig, get_config
from agent.api.message import Message


class Cortext(CustomLLM):
    """
    A custom language model implementation that extends CustomLLM.

    This class provides methods for text completion and embedding, integrating with
    the llama_index library and custom configurations.

    Attributes:
        mind_config (MindConfig): Configuration for the language model.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Cortext instance.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent class.
        """
        super().__init__(**self.mind_config.model_dump(), **kwargs)
        Settings.llm = self

        # embed_model = HuggingFaceEmbedding(
        #     model_name=self.mind_config.embed_model_name,
        #     device="cpu",
        # )
        # Settings.embed_model = embed_model

    @property
    def mind_config(self) -> MindConfig:
        """
        Get the mind configuration.

        Returns:
            MindConfig: The current mind configuration.
        """
        return get_config().mind

    @property
    def metadata(self) -> LLMMetadata:
        """
        Get metadata about the language model.

        Returns:
            LLMMetadata: Metadata including context window, output length, and model name.
        """
        return LLMMetadata(
            context_window=self.mind_config.context_window,
            num_output=self.mind_config.max_tokens,
            model_name=self.mind_config.model_name,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Generate a chat response for the given messages.
        """
        tools: list[dict[str, Any]] = []
        if "tools" in kwargs:
            tools_from_kwargs = kwargs.get("tools")
            if isinstance(tools_from_kwargs, list):
                tools = tools_from_kwargs
        converted_messages: list[Message] = [Message.from_chat_message(message) for message in messages]

        chat_completion_response = get_chat_completion(
            messages=converted_messages,
            mind_config=self.mind_config,
            functions_schema=tools
        )

        chat_response = ChatResponse(message=chat_completion_response.choices[0].message.to_chat_message())
        return chat_response

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (str): The input prompt for completion.
            **kwargs: Additional keyword arguments for text completion.

        Returns:
            CompletionResponse: The generated completion response.
        """
        print("complete prompt: ", prompt)
        tools: list[dict[str, Any]] = []
        if "tools" in kwargs:
            tools_from_kwargs = kwargs.get("tools")
            if isinstance(tools_from_kwargs, list):
                tools = tools_from_kwargs

        print(kwargs)

        return get_text_completion(
            prompt=prompt,
            mind_config=self.mind_config,
            tools=tools
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        """
        Generate a streaming completion for the given prompt.

        Args:
            prompt (str): The input prompt for completion.
            formatted (bool): Whether the output should be formatted (unused in this implementation).
            **kwargs: Additional keyword arguments for text completion.

        Yields:
            CompletionResponseGen: A generator yielding the completion response.
        """

        tools: list[dict[str, Any]] = []
        if "tools" in kwargs:
            tools_from_kwargs = kwargs.get("tools")
            if isinstance(tools_from_kwargs, list):
                tools = tools_from_kwargs

        print("stream_complete prompt: ", prompt)
        completion_response = get_text_completion(
            prompt=prompt,
            mind_config=self.mind_config,
            tools=tools
        )
        yield completion_response
