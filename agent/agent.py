"""Agent module"""

from __future__ import annotations

import asyncio
import json
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

        async def _loop(agent: Agent):
            if agent.status == AgentStatus.AWAITING_INPUT:
                message = await agent.state.interface.get_input()
                if message is not None and isinstance(message, Event):
                    agent.hippocampus.append_to_history(message)
                    await agent.state.interface.show_output(message)
                else:
                    agent.status = AgentStatus.STANDBY
                    return

            new_event = await agent.run()
            await agent.process(new_event)

        while self.can_act:
            await _loop(self)

        if not self.can_act:
            message = Event(role="system", content="Returning to human control.")
            await self.state.interface.show_output(message)
            self.status = AgentStatus.AWAITING_INPUT
            self.state.step_number = 0
            await self.loop()
        else:
            self.status = AgentStatus.STANDBY
            await self.state.interface.exit_program()

    async def run(self) -> Event:
        """
        Run the agent.

        The agent's memory is used to create a prompt for a large language model.
        The model can either:
        - Return a message to the user
        - Call a tool
        """
        self.status = AgentStatus.PROCESSING
        inference_config = {
            "prompt": self.hippocampus.events,
            "structure": self.tool_schemas,
            "tool_names": list(self.state.tools_map.keys()),
            "output_type": FunctionCall,
            "add_reminders": False,
            "add_generation_prompt": True,
            "prefill": None,
            "temperature": 1.1,
            "min_p": 0.05,
            "min_tokens_to_keep": 10,
        }
        buffer = ""
        structured_output = ""
        # render live updates off main thread so llm can output faster
        for output in self.inference(**inference_config):
            decoded_output = self.inference.front_end.tokenizer.decode(output.token_ids)
            if self.inference.engine.is_within_value:
                structured_output += decoded_output
            else:
                buffer += decoded_output
            self.state.live_render((buffer, structured_output))

        await self.state.interface.end_live_output()
        tool_calls = []
        if self.inference.engine.has_reached_accept_state:
            engine_output = list(self.inference.engine.read_output(FunctionCall))
            self.inference.engine.reset()
            for output in engine_output:
                buffer = output.buffer
                tool_use = ToolUse("function", output.value)
                tool_calls.append(tool_use)
                break

        self.status = AgentStatus.SUCCESS
        return Event(
            state=EventState.ASSISTANT,
            content=buffer,
            tool_calls=tool_calls,
        )

    async def process(self, event: Event) -> None:
        """
        Process an event.

        This method handles the event, appends it to the history, and processes
        any tool calls.
        """
        await self.state.interface.show_output(event)
        self.hippocampus.append_to_history(event)
        self.state.step_number += 1
        for tool_called in event.tool_calls:
            with self.state.interface.console.status(f"[bold yellow]Using {tool_called.function.name} tool"):
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
    async def create(interface: Interface, inference: LocalInference | None = None) -> Agent:
        """
        Create an agent.
        """
        await interface.clear()
        agent_name = await Agent.get_agent_name()
        system_prompt = await Agent.get_agent_prompt()
        if inference is None:
            model_path = await Agent.get_model_path()
            with interface.console.status("[bold cyan]Loading model..."):
                inference = LocalInference(model_path)
        return Agent(interface, inference, name=agent_name, system_prompt=system_prompt)

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
    async def get_agent_prompt() -> str:
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
    def tool_schemas(self) -> list[dict]:
        """
        Get the tool schemas.
        """
        return [tool.get_invocation_schema() for tool in self.state.tools_map.values()]

    @property
    def system_prompt(self) -> Event:
        prompt = self.state.system_prompt
        prompt += "\n\n---- Tools ----\n"
        for tool in self.state.tools_map.values():
            schema = tool.json_schema.get('function', tool.json_schema.get('schema', {}))
            prompt += f"Tool name: {tool.name}\n"
            prompt += f'Tool description: \n"""\n{tool.description or "name indicates function."}\n"""\n'
            prompt += f"Tool schema: \n{json.dumps(schema, indent=2)}\n"
        prompt += "\n---- End of tools ----\n"
        control_tokens = self.inference.front_end.tokenizer.control_tokens
        tool_use_token_start = control_tokens.tool_use_token_start
        tool_use_token_end = control_tokens.tool_use_token_end
        prompt += f'Starting delimiter: "{tool_use_token_start}"\n'
        prompt += f'Ending delimiter: "{tool_use_token_end}".\n'
        prompt += f'Wrap tool calls between the delimiters "{tool_use_token_start}" and "{tool_use_token_end}".\n'
        prompt += "Any other text is yours to use as you see fit and is not shown to the user.\n"
        prompt += "Your tools give you agency.\n"
        return Event(role="system", content=prompt)

    @property
    def can_act(self) -> bool:
        """
        The agent can act if the current step number is less than or equal to the
        maximum number of sub-steps.

        Step count is reset to 0 when the agent returns to human control.
        """
        return (
            self.state.step_number <= MAX_SUB_STEPS
            and self.status not in [AgentStatus.STANDBY, AgentStatus.FAILED]
        )

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
        tools: list[Tool] | None = None,
    ):
        self.interface = interface
        self.name = name
        self.seed = seed or randint(0, 1000000)
        self.step_number = 0
        self.system_prompt = system_prompt
        self.tools_map = {tool.name: tool for tool in tools or Tool.load()}
        self.render_tasks: set[asyncio.Task] = set()

    def live_render(self, output: tuple[str, str]):
        """
        Render live output, semantically a tuple of (buffer, structured_output)
        """
        task = asyncio.create_task(self.interface.show_live_output(output))
        self.render_tasks.add(task)
        task.add_done_callback(lambda t: self.render_tasks.remove(t))

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
