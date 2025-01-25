import logging
import time
from typing import Any

from agent.event import Event, State
from agent.interface import CLIInterface, Interface
from agent.model_inference.front_ends import FrontEnd, FrontEndType
from tools import FunctionCall, Tool, ToolCall

logger = logging.getLogger(__name__)


class LocalInference:
    def __init__(
        self,
        model_path: str,
        front_end_type: FrontEndType | None = None,
        interface: Interface | None = None,
    ):
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
        self.interface = interface or CLIInterface()

    async def __call__(
        self,
        prompt: str | list[dict[str, Any]],
        tools: list[Tool],
        tool_calls: dict[str, list[dict[str, Any]]],
        schema: dict[str, Any] | None = None,
        prefill: str | None = None,
        add_generation_prompt: bool = True,
        add_reminders: bool = True,
        continue_message_id: str | None = None,
        **kwargs,
    ) -> list[Event]:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (str): The input prompt for completion.
            tools (list[Tool]): List of available tools.
            schema (dict[str, Any] | None): JSON schema to constrain the output.
            prefill (str | None): Prefill text to add to the prompt.
            add_generation_prompt (bool): Whether to add the generation prompt.
            add_reminders (bool): Whether to add reminders.
            continue_message_id (str | None): The ID of the message to continue from.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Message]: The generated messages.
        """
        tic = time.perf_counter()
        structured_schema = schema or [tool.get_invocation_schema() for tool in tools]
        delimiters = self.front_end.control_tokens.tool_use_delimiters()
        self.front_end.engine.configure(
            structured_schema,
            delimiters=delimiters,
            min_scratchpad_length=0,
        )
        toc = time.perf_counter()
        setup_time = toc - tic
        logger.debug(f"Engine setup time: {setup_time:.4f}s")

        tokenizer_config = {
            "prompt": prompt,
            "tools": [tool.to_dict() for tool in tools],
            "tool_calls": tool_calls,
            "add_generation_prompt": add_generation_prompt,
            "add_reminders": add_reminders,
            "continue_message_id": continue_message_id,
            "model_type": self.front_end.model_type,
            **self.front_end.control_tokens.model_dump(),
        }
        encoded_prompt = self.front_end.tokenizer.encode(**tokenizer_config)
        assert isinstance(encoded_prompt, list)

        if prefill:
            encoded_prefill = self.front_end.tokenizer.encode(
                prefill, add_special_tokens=False
            )
            assert isinstance(encoded_prefill, list)
            encoded_prompt.extend(encoded_prefill)

        logger.info(self.front_end.tokenizer.decode(encoded_prompt))
        if continue_message_id:
            logger.debug(f"Continuing message {continue_message_id}")

        results: list[FrontEnd.Result] = []
        tic = time.perf_counter()
        for n, result in enumerate(
            self.front_end(
                encoded_prompt,
                previous_prompt_tokens=self.previous_prompt_tokens,
                **kwargs,
            )
        ):
            results.append(result)
            encoded_prompt.extend(result.token_ids)

            if n == 0:
                self.previous_prompt_tokens = encoded_prompt
                toc = time.perf_counter()
                prompt_time = toc - tic
                logger.debug(f"Prompt time: {prompt_time:.4f}s")
            else:
                self.previous_prompt_tokens.extend(result.token_ids)

            decoded_text = self.front_end.tokenizer.decode(result.token_ids)
            has_end_tokens = any(
                token in decoded_text
                for token in self.front_end.control_tokens.end_tokens()
            )
            if not has_end_tokens:
                await self.interface.show_live_output(decoded_text)

            if has_end_tokens or result.schema_complete:
                await self.interface.end_live_output()
                break

        messages = []
        if results and results[-1].schema_complete:
            for engine_output in self.front_end.engine.read_output(FunctionCall):
                scratch_pad = engine_output.scratchpad
                inner_value = engine_output.value
                tool_call = ToolCall("function", inner_value)
                messages.append(
                    self.create_message(
                        scratch_pad,
                        tool_call,
                        continue_message_id,
                    )
                )
            self.front_end.engine.reset()
        else:
            new_tokens = [
                token_id for result in results for token_id in result.token_ids
            ]
            generated_text = self.front_end.tokenizer.decode(new_tokens)
            messages.append(self.create_message(generated_text))

        toc = time.perf_counter()
        generation_time = toc - tic
        logger.debug(f"Generation time: {generation_time:.4f}s")

        return messages

    @staticmethod
    def create_message(
        scratch_pad: str,
        tool_calls: list[ToolCall] | ToolCall | None = None,
        continue_message_id: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> Event:
        """Create a Message object based on the presence of tool calls."""
        if tool_calls:
            return Event(
                event_id=continue_message_id,
                role="assistant",
                content=scratch_pad,
                tool_calls=tool_calls if isinstance(tool_calls, list) else [tool_calls],
                name="scratch pad",
                state=State.TOOL_CALL,
                metadata=usage,
            )
        return Event(
            event_id=continue_message_id,
            role="assistant",
            content=scratch_pad,
            state=State.ASSISTANT_RESPONSE,
            name="response",
            metadata=usage,
        )
