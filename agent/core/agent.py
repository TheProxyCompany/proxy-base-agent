"""Agent module"""

from __future__ import annotations

import logging
import uuid
from enum import Enum
from random import randint
from typing import TypeVar

from agent.core.hippocampus import Hippocampus
from agent.core.interaction import Interaction
from agent.interface import CLIInterface, Interface
from agent.llm.local import LocalInference
from agent.llm.models import get_available_models
from agent.prompts import get_available_prompts, load_prompt
from agent.tools import Tool, ToolCall

logger = logging.getLogger(__name__)

MAX_SUB_STEPS: int = 10

T = TypeVar("T")


class Agent:
    class Status(Enum):
        # Core System States
        PROCESSING = "processing"
        STANDBY = "standby"
        IDLE = "idle"

        # Error and Recovery States
        SUCCESS = "success"
        FAILED = "failed"
        RETRYING = "retrying"
        BLOCKED = "blocked"

        def __str__(self):
            return self.value

        @classmethod
        def from_string(cls, state_string: str):
            return cls(state_string)

    def __init__(
        self,
        name: str,
        interface: Interface,
        inference: LocalInference,
        system_prompt_name: str | None,
        seed: int | None = None,
        tools: list[Tool] | list[str] | None = None,
        **inference_kwargs,
    ):
        """Initialize an agent."""
        self.seed = seed or randint(0, 1000000)
        self.name = name
        self.step_number = 0
        self.status = Agent.Status.IDLE

        self.system_prompt_name = system_prompt_name
        self.inference_kwargs = inference_kwargs

        self.tools: dict[str, Tool] = {}
        for tool in tools or Tool.load():
            if isinstance(tool, Tool):
                self.tools[tool.name] = tool
            elif isinstance(tool, str):
                self.tools[tool] = Tool.load(file_name=tool)[0]

        self.inference = inference
        self.interface = interface
        self.hippocampus = Hippocampus(self.system_prompt)

    @property
    def can_act(self) -> bool:
        """
        The agent can act if the current step number is less than or equal to the
        maximum number of sub-steps.

        Step count is reset to 0 when the agent returns to human control.
        """
        return self.step_number <= MAX_SUB_STEPS and self.status not in [
            Agent.Status.STANDBY,
            Agent.Status.SUCCESS,
        ]

    async def loop(self) -> None:
        """
        Run the agent within a loop.

        The agent will take action until it reaches the maximum number of sub-steps.
        """
        message = await self.interface.get_input(
            message="Enter your message [enter to send, Ctrl+C to exit]:",
            qmark=">",
        )
        if isinstance(message, Interaction):
            self.status = Agent.Status.PROCESSING
            if message.content:
                self.hippocampus.append_to_history(message)
                await self.interface.show_output(message)
        elif message is None:
            self.status = Agent.Status.STANDBY
            await self.interface.exit_program()
            return

        self.step_number = 0
        while self.can_act:
            self.step_number += 1
            self.status = Agent.Status.PROCESSING
            scratchpad, tool_call = await self.output(ToolCall)
            await self.take_action(scratchpad, tool_call)

        await self.loop()

    async def output(
        self,
        output_type: type[T],
        prompt: list[Interaction] | None = None,
        tools: list[Tool] | None = None,
    ) -> tuple[str, T | None]:
        """
        Use a language model to generate the agent's next output.
        Accepts a prompt and tools to use, and an expected output type.
        """

        buffer: list[int] = []
        structured: list[int] = []
        inference_config = {
            "prompt": [e.to_dict() for e in prompt or self.hippocampus.events.values()],
            "structure": [t.to_dict() for t in tools or self.tools.values()],
            "output_type": output_type,
            "system_reminder": self.system_reminder,
            **self.inference_kwargs,
        }
        for cast_output in self.inference(**inference_config):
            if self.inference.engine.is_within_value:
                structured.extend(cast_output.token_ids)
            else:
                buffer.extend(cast_output.token_ids)
            decoded_output = self.inference.engine.tokenizer.decode(buffer + structured)
            self.interface.show_live_output(decoded_output)

        self.interface.end_live_output()
        scratchpad = self.inference.engine.tokenizer.decode(buffer)
        structured_output = self.inference.engine.tokenizer.decode(structured)
        cast_output = self.inference.engine.cast_output(structured_output, output_type)
        logger.debug(f"Scratchpad: {scratchpad}")
        logger.debug(f"Structured Output: {cast_output or structured_output}")
        return scratchpad, cast_output

    async def take_action(self, scratchpad: str, tool_call: ToolCall | None) -> None:
        """
        Take actions based on an event.

        This method handles the event, appends it to the history, and processes
        any tool calls.
        """
        action = Interaction(
            role=Interaction.Role.ASSISTANT,
            name=self.name,
            scratchpad=scratchpad,
        )
        if tool_call:
            action.metadata["tool_call"] = tool_call
            action.metadata["tool_result"] = self.use_tool(tool_call)
        elif not tool_call and scratchpad:
            tool_call = self.get_thinking_tool(scratchpad)
            action.metadata["tool_call"] = tool_call
            del action.metadata["scratchpad"]

        if not tool_call:
            raise ValueError("No tool call or scratch pad provided")

        action.metadata["tool_result"] = self.use_tool(tool_call)

        await self.interface.show_output(action)
        self.hippocampus.append_to_history(action)

    def use_tool(self, tool_call: ToolCall) -> Interaction:
        """Use a tool and return results.

        Args:
            tool_use: The tool call to use.
        """
        try:
            tool = self.tools[tool_call.name]
            with self.interface.console.status(f"[yellow]Using {tool_call.name}"):
                result = tool(self, **tool_call.arguments)
            assert isinstance(result, Interaction)
            return result
        except Exception as e:
            self.status = Agent.Status.FAILED
            return Interaction(
                role=Interaction.Role.TOOL,
                content=f"Tool call failed: {e}",
            )

    @staticmethod
    async def create(
        interface: Interface | None = None,
        inference: LocalInference | None = None,
        **inference_kwargs,
    ) -> Agent:
        """
        Create an agent.

        Args:
            interface: Interface for I/O operations
            inference: Inference engine
            inference_kwargs: kwargs used when inferencing the agent
        """
        interface = interface or CLIInterface()
        await interface.clear()
        if inference is None:
            model_path = await Agent.get_model_path(interface)
            with interface.console.status("[yellow]Loading model..."):
                inference = LocalInference(model_path)
        agent_name = await Agent.get_agent_name(interface)
        system_prompt = await Agent.get_agent_prompt(interface)
        return Agent(
            agent_name,
            interface,
            inference,
            system_prompt_name=system_prompt,
            **inference_kwargs,
        )

    @staticmethod
    async def get_agent_name(interface: Interface) -> str:
        """
        Prompt the user for an agent name.

        Returns:
            str: The chosen agent name or a generated UUID if left blank.
        """
        response = await interface.get_input(
            message="Give the ai a name:",
            default="Cerebra",
        )
        agent_name = response.content if isinstance(response, Interaction) else response
        assert isinstance(agent_name, str)
        final_name = agent_name.strip() if agent_name else f"agent_{uuid.uuid4()}"
        return final_name

    @staticmethod
    async def get_agent_prompt(interface: Interface) -> str:
        """
        Prompt the user for an agent prompt.

        Returns:
            str: The chosen agent prompt.
        """
        available_prompts = [file.split(".txt")[0] for file in get_available_prompts()]
        if len(available_prompts) == 1:
            return available_prompts[0]

        prompt_name = await interface.get_input(
            message="Select a prompt for the agent:",
            choices=available_prompts,
            default="base" if "base" in available_prompts else available_prompts[0],
        )
        return (
            prompt_name.content if isinstance(prompt_name, Interaction) else prompt_name
        )

    @staticmethod
    async def get_model_path(interface: Interface) -> str:
        """
        Prompt the user for a model path.

        Returns:
            str: The chosen model name.
        """
        available_models: dict[str, str] = {
            name: path for name, path, _ in get_available_models()
        }
        if not available_models:
            huggingface_model_name = await interface.get_input(
                message="Download a model from HuggingFace:",
                default="mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
            )
            return huggingface_model_name.content

        model_name = await interface.get_input(
            message="Select a model for the agent:",
            choices=list(available_models.keys()),
            default=next(iter(available_models.keys())),
        )
        model_path = available_models[model_name.content]
        return model_path

    @property
    def system_prompt(self) -> Interaction:
        prompt = load_prompt(self.system_prompt_name)
        if prompt is None:
            prompt = f"No System Prompt Found for {self.system_prompt_name}"
        else:
            try:
                prompt = prompt.format(
                    name=self.name,
                    tool_use_instructions=self.tool_use_instructions,
                    tool_list=self.tool_list,
                )
            except Exception:
                pass

        return Interaction(role=Interaction.Role.SYSTEM, content=prompt)

    @property
    def tool_list(self) -> str:
        """
        List of tools available to the agent.
        """
        prompt = "---- Tool Definitions ----\n"
        for tool in self.tools.values():
            prompt += f"\n{tool}"
        return prompt

    @property
    def tool_use_instructions(self) -> str:
        """
        Instructions on how to use tools.
        """
        prompt = f"Standardized tool call schema:\n{ToolCall.invocation_schema()}\n"
        prompt += f"Available tools: [{', '.join(self.tools.keys())}]\n"
        if delimiters := self.inference.front_end.tokenizer.control_tokens.tool_use_delimiters():
            prompt += "You MUST use these delimiters to separate tool use from your scratch pad.\n"
            prompt += f"Start of tool use delimiter: {delimiters[0]!r}\n"
            prompt += f"End of tool use delimiter: {delimiters[1]!r}\n"
        return prompt

    @property
    def system_reminder(self) -> dict | None:
        if self.step_number % 5 != 0:
            return None
        reminder = self.tool_use_instructions
        reminder += "Maintain your sense of self and your place in the conversation.\n"
        reminder += "Do not get distracted. Pause, think, and then act.\n"
        reminder += "Stay focused on the present interaction.\n"
        if (
            delimiters
            := self.inference.front_end.tokenizer.control_tokens.tool_use_delimiters()
        ):
            reminder += "You MUST use these delimiters to separate tool use from your scratch pad.\n"
            reminder += f"Start of tool use delimiter: {delimiters[0]!r}\n"
            reminder += f"End of tool use delimiter: {delimiters[1]!r}\n"
        return Interaction(content=reminder, role=Interaction.Role.SYSTEM).to_dict()

    def get_thinking_tool(self, scratchpad: str) -> ToolCall:
        return ToolCall(name="internal_thoughts", arguments={"thoughts": scratchpad})

    def __repr__(self) -> str:
        return f"{self.name} ({self.status})"
