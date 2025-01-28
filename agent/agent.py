"""Agent module"""

from __future__ import annotations

import logging
import uuid
from enum import Enum
from random import randint

from memory.hippocampus import Hippocampus

from agent.inference import (
    DEFAULT_MODEL_FOLDER,
    DEFAULT_MODEL_NAME,
    get_available_models,
)
from agent.inference.inference.local import LocalInference
from agent.inference.prompts import get_available_prompts, load_prompt_template
from agent.interaction import Interaction
from interface import CLIInterface, Interface
from tools import FunctionCall, Tool, ToolUse

logger = logging.getLogger(__name__)

MAX_SUB_STEPS: int = 10


class Agent:

    class Status(Enum):
        # Core System States
        PROCESSING = "processing"
        STANDBY = "standby"
        AWAITING_INPUT = "awaiting_input"

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
        self.inference = inference
        self.interface = interface
        self.hippocampus = Hippocampus()

        self.seed = seed or randint(0, 1000000)
        self.name = name
        self.step_number = 0
        self.status = Agent.Status.AWAITING_INPUT

        self.system_prompt_name = system_prompt_name
        self.inference_kwargs = inference_kwargs

        self.tools: dict[str, Tool] = {}
        for tool in tools or Tool.load():
            if isinstance(tool, Tool):
                self.tools[tool.name] = tool
            elif isinstance(tool, str):
                self.tools[tool] = Tool.load(file_name=tool)[0]

    async def loop(self) -> None:
        """
        Run the agent within a loop.

        The agent will take action until it reaches the maximum number of sub-steps.
        """
        message = await self.interface.get_input(
            message="Enter your message [enter to send, Ctrl+C to exit]:",
            qmark=">",
            default="",
        )
        if isinstance(message, Interaction):
            self.status = Agent.Status.PROCESSING
            self.hippocampus.append_to_history(message)
            await self.interface.show_output(message)
        elif message is None:
            self.status = Agent.Status.STANDBY
            await self.interface.exit_program()
            return

        self.step_number = 0
        while self.can_act:
            new_event = await self.run()
            await self.take_actions(new_event)

        await self.loop()

    async def run(self) -> Interaction:
        """
        Use a language model to generate the agent's next output.
        """
        self.status = Agent.Status.PROCESSING
        inference_config = {
            "prompt": [event.to_dict() for event in self.hippocampus.events.values()],
            "structure": [tool.to_dict() for tool in self.tools.values()],
            "output_type": FunctionCall,
            **self.inference_kwargs,
        }
        buffer = ""
        structured_output = ""
        # render live updates off main thread so llm can output faster
        for output in self.inference(**inference_config):
            decoded_output = self.inference.front_end.tokenizer.decode(output.token_ids)
            if (
                self.inference.engine.is_within_value
                or self.inference.engine.has_reached_accept_state
            ):
                structured_output += decoded_output
                logger.debug(f"Structured: {structured_output}")
            else:
                buffer += decoded_output
                logger.debug(f"Buffer: {buffer}")
            await self.interface.show_live_output((buffer, structured_output))

        await self.interface.end_live_output()
        breakpoint()
        self.status = Agent.Status.SUCCESS

        tool_calls = []
        if self.inference.engine.in_accepted_state:
            engine_output = list(self.inference.engine.output(FunctionCall))
            for output in engine_output:
                buffer = output.buffer
                tool_use = ToolUse("function", output.value)
                tool_calls.append(tool_use)
                break

        return Interaction(
            role=Interaction.Role.ASSISTANT,
            content=buffer,
            tool_calls=tool_calls,
        )

    async def take_actions(self, event: Interaction) -> None:
        """
        Take actions based on an event.

        This method handles the event, appends it to the history, and processes
        any tool calls.
        """
        await self.interface.show_output(event)
        self.hippocampus.append_to_history(event)
        self.step_number += 1
        for tool_called in event.tool_calls:
            with self.interface.console.status(
                f"[bold yellow]Using {tool_called.function.name} tool"
            ):
                tool_result = self.use_tool(tool_called)
            self.hippocampus.append_to_history(tool_result)
            await self.interface.show_output(tool_result)

    def use_tool(self, tool_use: ToolUse) -> Interaction:
        """Use a tool and return results.

        Args:
            tool_use: The tool call to use.
        """

        content = ""
        try:
            tool = self.tools[tool_use.function.name]
            result = tool(self, **tool_use.function.arguments)
            if isinstance(result, Interaction):
                result.event_id = tool_use.tool_use_id
                return result

            content = str(result)
        except Exception as e:
            content = str(e)
            self.status = Agent.Status.FAILED

        return Interaction(
            event_id=tool_use.tool_use_id,
            role=Interaction.Role.TOOL,
            content=content,
            name=tool_use.function.name,
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
        if inference is None:
            model_path = await Agent.get_model_path(interface)
            with interface.console.status("[bold cyan]Loading model..."):
                inference = LocalInference(model_path)

        await interface.clear()
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
            message="Enter a name (hit enter for Cerebra):",
            default="Cerebra"
        )
        agent_name = response.content if isinstance(response, Interaction) else response
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
        prompt_name = await interface.get_input(
            message="Select a prompt for the agent (hit enter for default):",
            choices=available_prompts,
            default="base" if "base" in available_prompts else available_prompts[0],
        )
        return prompt_name.content if isinstance(prompt_name, Interaction) else prompt_name

    @staticmethod
    async def get_model_path(interface: Interface) -> str:
        """
        Prompt the user for a model path.

        Returns:
            str: The chosen model name.
        """
        available_models = [name for name, _ in get_available_models()]
        model_name = await interface.get_input(
            message="Select a model for the agent (hit enter for default):",
            choices=available_models,
            default=DEFAULT_MODEL_NAME,
        )
        model_path = f"{DEFAULT_MODEL_FOLDER}/{model_name.content}"
        return model_path

    @property
    def system_prompt(self) -> Interaction:
        prompt = load_prompt_template(self.system_prompt_name)
        if prompt is None:
            prompt = f"No System Prompt Found for {self.system_prompt_name}"
        else:
            try:
                prompt = prompt.format(
                    tool_use_instructions=self.tool_use_instructions,
                    tool_list=self.tool_list,
                )
            except Exception:
                pass

        return Interaction(
            role=Interaction.Role.SYSTEM,
            content=prompt,
        )

    @property
    def tool_list(self) -> str:
        """
        List of tools available to the agent.
        """
        prompt = "---- Tool List ----"
        for tool in self.tools.values():
            prompt += f"\n{tool}"
        return prompt

    @property
    def tool_use_instructions(self) -> str:
        """
        Instructions on how to use tools.
        """
        prompt = f"Invoke a tool with the following schema:\n{FunctionCall.invocation_schema()}\n"
        if (
            delimiters
            := self.inference.front_end.tokenizer.control_tokens.tool_use_delimiters()
        ):
            open_delim = delimiters[0].replace("\n", "\\n")
            close_delim = delimiters[1].replace("\n", "\\n")
            prompt += f'Use the delimiters "{open_delim}" and "{close_delim}"'
            prompt += " to separate tool use from the rest of your output."
        return prompt

    @property
    def tool_reminder(self) -> str:
        """
        Reminders for the agent.
        """
        reminder = "\n" + self.tool_use_instructions + "\n"
        reminder += f"[Available tools]: {', '.join(self.tools.keys())}\n"
        return reminder

    @property
    def can_act(self) -> bool:
        """
        The agent can act if the current step number is less than or equal to the
        maximum number of sub-steps.

        Step count is reset to 0 when the agent returns to human control.
        """
        return self.step_number <= MAX_SUB_STEPS and self.status not in [
            Agent.Status.STANDBY,
            Agent.Status.FAILED,
            Agent.Status.AWAITING_INPUT,
        ]

    def __repr__(self) -> str:
        return f"{self.name} ({self.status})"
