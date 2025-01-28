"""Agent module"""

from __future__ import annotations

import asyncio
import logging
import uuid
from enum import Enum
from random import randint

import questionary

from agent.event import Event, EventState
from agent.model_inference.inference.local import LocalInference
from agent.prompts import get_available_prompts, load_prompt_template
from interface import Interface
from tools import FunctionCall, Tool, ToolUse

logger = logging.getLogger(__name__)

MAX_SUB_STEPS: int = 10


class Agent:
    def __init__(self, interface: Interface, inference: LocalInference, **kwargs):
        """Initialize an agent."""
        from memory.hippocampus import Hippocampus

        self.inference = inference
        self.state = AgentState(interface, **kwargs)
        self.hippocampus = Hippocampus(self.state)
        self.hippocampus.append_to_history(self.system_prompt)
        self.status = AgentStatus.AWAITING_INPUT

    async def __call__(self) -> None:
        await self.loop()

    async def loop(self) -> None:
        """
        Run the agent within a loop.

        The agent will take action until it reaches the maximum number of sub-steps.
        """

        while self.can_act:
            await self.perception()
            if self.status == AgentStatus.STANDBY:
                return
            new_event = await self.act()
            await self.take_actions(new_event)

        if not self.can_act and self.status != AgentStatus.STANDBY:
            message = Event(role="system", content="Returning to human control.")
            await self.state.interface.show_output(message)
            self.status = AgentStatus.AWAITING_INPUT
            self.state.step_number = 0
            await self.loop()
        else:
            self.status = AgentStatus.STANDBY
            await self.state.interface.exit_program()

    async def perception(self) -> None:
        """
        Take input from the user (if needed)
        """
        if self.status != AgentStatus.AWAITING_INPUT:
            return

        message = await self.state.interface.get_input()
        if message and isinstance(message, Event):
            self.hippocampus.append_to_history(message)
            await self.state.interface.show_output(message)
        elif message is None:
            self.status = AgentStatus.STANDBY
            return

    async def act(self) -> Event:
        """
        Use a language model to generate the agent's next output.
        """
        self.status = AgentStatus.PROCESSING
        inference_config = {
            "prompt": [event.to_dict() for event in self.hippocampus.events.values()],
            "structure": [tool.to_dict() for tool in self.state.tools_map.values()],
            "system_reminder": self.system_reminder,
            "output_type": FunctionCall,
            "add_generation_prompt": True,
            "prefill": None,
            **self.state.inference_kwargs,
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
            await self.state.live_render((buffer, structured_output))

        await self.state.interface.end_live_output()
        breakpoint()
        self.status = AgentStatus.SUCCESS

        tool_calls = []
        if self.inference.engine.in_accepted_state:
            engine_output = list(self.inference.engine.output(FunctionCall))
            for output in engine_output:
                buffer = output.buffer
                tool_use = ToolUse("function", output.value)
                tool_calls.append(tool_use)
                break

        return Event(
            state=EventState.ASSISTANT,
            content=buffer,
            tool_calls=tool_calls,
        )

    async def take_actions(self, event: Event) -> None:
        """
        Take actions based on an event.

        This method handles the event, appends it to the history, and processes
        any tool calls.
        """
        await self.state.interface.show_output(event)
        self.hippocampus.append_to_history(event)
        self.state.step_number += 1
        for tool_called in event.tool_calls:
            with self.state.interface.console.status(
                f"[bold yellow]Using {tool_called.function.name} tool"
            ):
                tool_result = self.use_tool(tool_called)
            self.hippocampus.append_to_history(tool_result)
            await self.state.interface.show_output(tool_result)

    def use_tool(self, tool_use: ToolUse) -> Event:
        """Use a tool and return results.

        Args:
            tool_use: The tool call to use.
        """

        content = ""
        try:
            tool = self.state.tools_map[tool_use.function.name]
            result = tool(self, **tool_use.function.arguments)
            if isinstance(result, Event):
                result.event_id = tool_use.tool_use_id
                return result

            content = str(result)
        except Exception as e:
            content = str(e)
            self.status = AgentStatus.FAILED

        return Event(
            event_id=tool_use.tool_use_id,
            state=EventState.TOOL,
            content=content,
            name=tool_use.function.name,
        )

    @staticmethod
    async def create(
        interface: Interface,
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
        await interface.clear()
        agent_name = await Agent.get_agent_name()
        system_prompt = await Agent.get_agent_prompt()
        if inference is None:
            model_path = await Agent.get_model_path()
            with interface.console.status("[bold cyan]Loading model..."):
                inference = LocalInference(model_path)
        return Agent(
            interface,
            inference,
            name=agent_name,
            system_prompt=system_prompt,
            **inference_kwargs,
        )

    @staticmethod
    async def get_agent_name() -> str:
        """
        Prompt the user for an agent name.

        Returns:
            str: The chosen agent name or a generated UUID if left blank.
        """
        agent_name: str = await questionary.text(
            "Enter a name (hit enter for Cerebra):", default="Cerebra"
        ).ask_async()
        final_name = agent_name.strip() if agent_name else f"agent_{uuid.uuid4()}"
        return final_name

    @staticmethod
    async def get_agent_prompt() -> str | None:
        """
        Prompt the user for an agent prompt.

        Returns:
            str: The chosen agent prompt.
        """
        available_prompts = [file.split(".txt")[0] for file in get_available_prompts()]
        prompt_name = await questionary.select(
            message="Select a prompt for the agent (hit enter for default):",
            choices=available_prompts,
            default="base" if "base" in available_prompts else available_prompts[0],
        ).ask_async()
        return load_prompt_template(prompt_name)

    @staticmethod
    async def get_model_path() -> str:
        """
        Prompt the user for a model path.

        Returns:
            str: The chosen model path.
        """
        MODEL_PATH = ".language_models/Llama-3.1-SuperNova-Lite"
        model_path = await questionary.text(
            "Enter a model path (hit enter for default):", default=MODEL_PATH
        ).ask_async()
        return model_path

    @property
    def system_prompt(self) -> Event:
        prompt = self.state.system_prompt

        try:
            prompt = prompt.format(
                tool_use_instructions=self.tool_use_instructions,
                tool_list=self.tool_list,
            )
        except Exception:
            pass

        return Event(role="system", content=prompt)

    @property
    def system_reminder(self) -> str:
        """
        Reminders for the agent.
        """
        reminder = ""
        reminder += self.tool_use_instructions
        reminder += f"[Available tools]: {', '.join(self.state.tools_map.keys())}\n"
        return reminder

    @property
    def tool_list(self) -> str:
        """
        List of tools available to the agent.
        """
        prompt = "---- Tool List ----"
        for tool in self.state.tools_map.values():
            prompt += f"\n{tool}"
        return prompt

    @property
    def tool_use_instructions(self) -> str:
        """
        Instructions on how to use tools.
        """
        prompt = f"Invoke a tool with the following schema:\n{FunctionCall.invocation_schema()}\n"
        if delimiters := self.inference.front_end.tokenizer.control_tokens.tool_use_delimiters():
            open_delim = delimiters[0].replace("\n", "\\n")
            close_delim = delimiters[1].replace("\n", "\\n")
            prompt += f'Use the delimiters "{open_delim}" and "{close_delim}"'
            prompt += " to separate tool use from the rest of your output."
        return prompt

    @property
    def can_act(self) -> bool:
        """
        The agent can act if the current step number is less than or equal to the
        maximum number of sub-steps.

        Step count is reset to 0 when the agent returns to human control.
        """
        return self.state.step_number <= MAX_SUB_STEPS and self.status not in [
            AgentStatus.STANDBY,
            AgentStatus.FAILED,
        ]

    def __repr__(self) -> str:
        return f"Agent({self.state})"


class AgentState:
    """Represents the state of an agent.

    Maintains the agent's core attributes and state information during execution.

    Attributes:
        interface: Interface for I/O operations
        name: Agent's identifier
        system_prompt: Base prompt that defines agent behavior
        seed: Random seed for reproducibility
        step_number: Current step in execution
        tools_map: Dictionary mapping tool names to Tool instances
    """

    def __init__(
        self,
        interface: Interface,
        name: str,
        system_prompt: str,
        seed: int | None = None,
        tools: list[Tool] | list[str] | None = None,
        **inference_kwargs,
    ):
        """
        Initialize an agent state.

        Args:
            interface: Interface for I/O operations
            name: Agent's identifier
            system_prompt: Base prompt that defines agent behavior
            seed: Random seed for reproducibility
            tools: List of tools to use
            inference_kwargs: Additional inference kwargs
        """
        self.interface = interface
        self.name = name
        self.seed = seed or randint(0, 1000000)
        self.step_number = 0
        self.system_prompt = system_prompt
        self.render_tasks: set[asyncio.Task] = set()
        self.inference_kwargs = inference_kwargs
        self.tools_map: dict[str, Tool] = {}
        for tool in tools or Tool.load():
            if isinstance(tool, Tool):
                self.tools_map[tool.name] = tool
            elif isinstance(tool, str):
                self.tools_map[tool] = Tool.load(file_name=tool)[0]

    async def live_render(self, output: tuple[str, str]):
        """
        Render live output, semantically a tuple of (buffer, structured_output)
        """
        await self.interface.show_live_output(output)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.name} (seed: {self.seed})"


class AgentStatus(Enum):
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
