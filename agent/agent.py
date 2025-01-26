"""Agent module"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from random import randint

import questionary

from agent.event import Event, State
from agent.interface import Interface
from agent.model_inference.local_inference import LocalInference
from agent.prompts import get_available_prompts, load_prompt_template
from tools import FunctionCall, Tool, ToolUse

logger = logging.getLogger(__name__)

MAX_SUB_STEPS: int = 10


class AgentState:
    """Represents the state of an agent.

    Maintains the agent's core attributes and state information during execution.

    Attributes:
        interface: Interface for I/O operations
        name: Agent's identifier
        system_prompt: Base prompt that defines agent behavior
        seed: Random seed for reproducibility
        step_number: Current step in execution
        continue_message_id: ID of message to continue from, if any
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
        self.current_event_id: str | None = None
        self.interface = interface
        self.name = name
        self.seed = seed or randint(0, 1000000)
        self.step_number = 0
        self.system_prompt = system_prompt
        self.tools_map = {tool.name: tool for tool in tools or Tool.load()}

    def __repr__(self) -> str:
        readable = {
            "name": self.name,
            "step_number": self.step_number,
            "current_event_id": self.current_event_id,
        }
        return str(readable)


class Agent:

    def __init__(self, interface: Interface, inference: LocalInference, **kwargs):
        """Initialize an agent."""
        from agent.memory.hippocampus import Hippocampus

        self.inference = inference
        self.state = AgentState(interface, **kwargs)
        self.hippocampus = Hippocampus(self.state)
        self.hippocampus.append_to_history(self.system_prompt)

    async def __call__(self) -> None:
        await self.loop()

    async def loop(self) -> None:
        """
        Run the agent within a loop.

        The agent will take action until it reaches the maximum number of sub-steps.
        """
        while self.can_act:
            if self.state.step_number == 0:
                message = await self.state.interface.get_input()
                if message is not None and isinstance(message, Event):
                    self.hippocampus.append_to_history(message)
                    await self.state.interface.show_output(message)
                elif message:
                    break

            await self.run()

        if not self.can_act:
            message = Event(role="system", content="Returning to human control.")
            await self.state.interface.show_output(message)
            self.state.step_number = 0
            self.state.current_event_id = None
            await self.loop()
        else:
            await self.state.interface.exit_program()

    async def run(self) -> Event | None:
        """
        Run the agent.

        The agent's memory is used to create a prompt for a large language model.
        The model can either:
        - Return a message to the user
        - Call a tool
        """
        inference_config = {
            "prompt": self.hippocampus.events,
            "structure": self.tool_schemas,
            "tool_names": list(self.state.tools_map.keys()),
            "output_type": FunctionCall,
            "add_reminders": False,
            "add_generation_prompt": True,
            "prefill": None,
            "event_id": self.state.current_event_id,
            "temperature": 1.1,
            "min_p": 0.05,
            "min_tokens_to_keep": 10,
        }
        buffer = ""
        structured_output = ""
        # render live updates off main thread so llm can output faster
        render_tasks = set()
        for output in self.inference(**inference_config):
            decoded_output = self.inference.front_end.tokenizer.decode(output.token_ids)
            if self.inference.engine.is_within_value:
                structured_output += decoded_output
            else:
                buffer += decoded_output

            task = asyncio.create_task(self.state.interface.show_live_output(decoded_output))
            render_tasks.add(task)
            task.add_done_callback(lambda t: render_tasks.remove(t))

        if self.inference.engine.has_reached_accept_state:
            engine_output = list(self.inference.engine.read_output(FunctionCall))
            self.inference.engine.reset()
            for output in engine_output:
                self.buffer = output.buffer
                tool_call = ToolUse("function", output.value)
                return Event(role="assistant", content=self.buffer, tool_calls=[tool_call])
        else:
            return Event(role="assistant", content=self.buffer)

    async def process(self, event: Event) -> None:
        await self.state.interface.show_output(event)
        self.hippocampus.append_to_history(event)
        self.state.step_number += 1
        self.state.current_event_id = event.id
        for tool_called in event.tool_calls:
            tool_result = self.use_tool(tool_called)
            self.hippocampus.append_to_history(tool_result)
            await self.state.interface.show_output(tool_result)

    @staticmethod
    async def create(
        interface: Interface,
        inference: LocalInference | None = None
    ) -> Agent:

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
            "Enter a name (hit enter for Cerebra):",
            default="Cerebra"
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
        return [tool.get_invocation_schema() for tool in self.state.tools_map.values()]

    @property
    def system_prompt(self) -> Event:
        prompt = self.state.system_prompt
        prompt += "\n\n---- Tools ----\n"
        for tool in self.state.tools_map.values():
            prompt += f"\n---{tool.name}---\n"
            prompt += f"Tool name: {tool.name}\n"
            prompt += f'Tool description: \n"""\n{tool.description}\n"""\n'
            prompt += f"Tool schema: \n{json.dumps(tool.schema, indent=2)}\n"
        prompt += "\n---- End of tools ----\n"
        control_tokens = self.inference.front_end.tokenizer.control_tokens
        tool_use_token_start = control_tokens.tool_use_token_start
        tool_use_token_end = control_tokens.tool_use_token_end
        prompt += f'Starting delimiter: "{tool_use_token_start}\\n"\n'
        prompt += f'Ending delimiter: "{tool_use_token_end}".\n'
        prompt += f'Wrap tool calls between the delimiters "{tool_use_token_start}\\n" and "\\n{tool_use_token_end}".\n'

        prompt += "Any other text is yours to use as you see fit and is not shown to the user.\n"
        prompt += "Only you can see the tool calls and their results (unless specified otherwise).\n"
        return Event(role="system", content=prompt)

    @property
    def can_act(self) -> bool:
        """
        The agent can act if the current step number is less than or equal to the
        maximum number of sub-steps.

        Step count is reset to 0 when the agent returns to human control.
        """
        return self.state.step_number <= MAX_SUB_STEPS

    def use_tool(self, tool_use: ToolUse) -> Event:
        """Handle a single function call, execute the function, and return results.

        Args:
            function_call: The function call to handle.

        Returns:
            A tuple containing:
            - The message with the function call result.
            - A boolean indicating if the function call failed.
        """

        try:
            tool = self.state.tools_map[tool_use.function.name]
            with self.state.interface.console.status(
                f"[bold yellow]Calling {tool.name}..."
            ):
                result = tool(self, **tool_use.function.arguments)
            if isinstance(result, Event):
                result.id = tool_use.tool_id
                return result
            return Event(
                role="ipython",
                content=str(result),
                name=tool_use.function.name,
                event_id=tool_use.tool_id,
                state=State.TOOL_RESULT,
            )
        except Exception as e:
            return Event(
                role="ipython",
                content=str(e),
                name=tool_use.function.name,
                event_id=tool_use.tool_id,
                state=State.TOOL_ERROR,
            )

    def __repr__(self) -> str:
        return f"Agent({self.state})"
