import logging
import time
from typing import Any

from model_inference.front_ends import FrontEnd, FrontEndType

from agent.message import Message, MessageState
from tools import Tool, ToolCall

logger = logging.getLogger(__name__)


def debugbold(*args, **kwargs):
    """Print debug messages in bold yellow."""
    print("\033[1;33m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


class ModelInference:
    def __init__(self, model_path: str, front_end_type: FrontEndType | None = None):
        """
        Initialize the Inference class.

        Args:
            model_path (str): Path to the model.

        This method sets up the necessary components for inference, including:
        - Loading the model configuration
        - Initializing the tokenizer and model
        - Setting up caches and data structures for efficient inference
        """
        self.front_end = FrontEnd.from_type(model_path, front_end_type)
        self.previous_prompt_tokens: list[int] = []

    def __call__(
        self,
        prompt: str | list[dict[str, Any]],
        tools: list[Tool],
        schema: dict[str, Any] | None = None,
        prefill: str | None = None,
        add_generation_prompt: bool = True,
        add_reminders: bool = True,
        continue_message_id: str | None = None,
        **kwargs,
    ) -> list[Message]:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (str): The input prompt for completion.
            max_tokens (int): Maximum number of tokens to generate.
            tools (Optional[List[Dict[str, str]]]): List of available tools.
            tool_schema (Optional[Dict]): JSON schema to constrain the output.
            min_p (float): Minimum probability threshold for min-p sampling.
            min_tokens_to_keep (int): Minimum number of tokens to keep in min-p sampling.
            temp (float): Temperature for sampling.
            seed (Optional[int]): Seed for the random number generator.
            cache_prompt (bool): Whether to cache the prompt for future use.
            context_window (int): The context window size.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[str, Dict]: The generated completion and usage.
        """

        tic = time.perf_counter()
        self.front_end.engine.configure(schema or [tool.get_invocation_schema() for tool in tools], True)
        toc = time.perf_counter()
        setup_time = toc - tic
        logger.debug(f"Engine setup time: {setup_time:.4f}s")

        tokenizer_config = {
            "prompt": prompt,
            "tools": tools,
            "tool_calls": self.get_tool_calls(prompt),
            "add_generation_prompt": add_generation_prompt,
            "add_reminders": add_reminders,
            "continue_message_id": continue_message_id,
            "model_type": self.front_end.model_type,
            **self.front_end.control_tokens.model_dump(),
        }
        encoded_prompt = self.front_end.tokenizer.encode(**tokenizer_config)
        assert isinstance(encoded_prompt, list)

        if prefill and isinstance(encoded_prompt, list):
            encoded_prefill = self.front_end.tokenizer.encode(prefill, add_special_tokens=False)
            assert isinstance(encoded_prefill, list)
            encoded_prompt.extend(encoded_prefill)

        logger.info(self.front_end.tokenizer.decode(encoded_prompt))
        if continue_message_id:
            logger.info(f"Continuing from message {continue_message_id}")
        results: list[FrontEnd.Result] = []

        tic = time.perf_counter()
        try:
            for n, result in enumerate(
                self.front_end(
                    encoded_prompt,
                    previous_prompt_tokens=self.previous_prompt_tokens,
                    **kwargs,
                )
            ):
                if n == 0:
                    self.previous_prompt_tokens = encoded_prompt
                    toc = time.perf_counter()
                    prompt_time = toc - tic
                    logger.debug(f"Prompt time: {prompt_time:.4f}s")
                else:
                    self.previous_prompt_tokens.extend(result.token_ids)

                if (
                    self.front_end.tokenizer.eos_token_id in result.token_ids
                    or self.front_end.tokenizer.eom_token_id in result.token_ids
                ):
                    break

                results.append(result)
                encoded_prompt.extend(result.token_ids)
                decoded_text = self.front_end.tokenizer.decode(result.token_ids)
                if logger.getEffectiveLevel() > logging.DEBUG:
                    debugbold(decoded_text, end="", flush=True)
                else:
                    logger.debug(f"Generated {len(results)} tokens.")

                if result.schema_complete:
                    messages = []
                    for value in self.front_end.engine.get_current_value():
                        if self.front_end.engine.is_encapsulated:
                            scratch_pad = value[0]
                            tool_calls = [ToolCall(**call) for call in value[1]]
                            messages.append(
                                self.create_message(scratch_pad, tool_calls)
                            )
                        elif isinstance(value, dict):
                            tool_calls = [ToolCall(**call) for call in value.values()]
                            messages.append(self.create_message("", tool_calls))

                    self.front_end.engine.reset()
                    return messages

        except KeyboardInterrupt:
            print()
            logger.info("User interrupted generation")
        except Exception as e:
            message = Message(role="system", content=f"Error during generation.\nError: {e}")
            return [message]

        new_tokens = [token_id for result in results for token_id in result.token_ids]
        generated_text = self.front_end.tokenizer.decode(new_tokens)
        toc = time.perf_counter()
        generation_time = toc - tic
        logger.debug(f"Generation time: {generation_time:.4f}s")

        # usage = {
        #     "completion_tokens": len(results),
        #     "generation_time": generation_time,
        #     "setup_time": setup_time,
        #     "total_tokens": len(encoded_prompt),
        #     "total_time": generation_time + setup_time,
        #     "tokens_per_second": len(encoded_prompt) / generation_time,
        # }

        return [self.create_message(generated_text)]

    @staticmethod
    def create_message(
        scratch_pad: str,
        tool_calls: list[ToolCall] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> Message:
        """Create a Message object based on the presence of tool calls."""
        if tool_calls:
            return Message(
                role="assistant",
                content=scratch_pad,
                tool_calls=tool_calls,
                name="scratch pad",
                state=MessageState.TOOL_CALL,
                metadata=usage,
            )
        return Message(
            role="assistant",
            content=scratch_pad,
            state=MessageState.ASSISTANT_RESPONSE,
            name="response",
            metadata=usage,
        )

    @staticmethod
    def get_tool_calls(prompt: str | list[dict[str, Any]]) -> dict[str, Any]:
        """
        Precompute a mapping from tool_call_id to message for O(1) lookups
        """
        tic = time.perf_counter()
        if not isinstance(prompt, list):
            return {}

        # First map tool_call_ids to their messages
        tool_messages = {m["tool_call_id"]: m for m in prompt if "tool_call_id" in m}

        # Then collect tool calls from assistant messages and map to their responses
        tool_calls = {
            call["id"]: tool_messages[call["id"]]
            for msg in prompt
            if msg.get("role") == "assistant" and msg.get("content")
            for call in msg.get("tool_calls", [])
            if call.get("id") in tool_messages
        }

        logger.debug(f"Preprocessing prompt time: {time.perf_counter() - tic:.4f}s")
        return tool_calls
