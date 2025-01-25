"""Agent module"""

from __future__ import annotations

import json
import logging
import uuid
from random import randint
from typing import Any

import questionary

from agent.event import Event, State
from agent.interface import Interface
from agent.model_inference.local_inference import LocalInference
from agent.prompts import get_available_prompts, load_prompt_template
from tools import Tool, ToolCall

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
        self.continue_message_id: str | None = None
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
            "continue_message_id": self.continue_message_id,
        }
        return str(readable)

class Agent:
    def __init__(self, interface: Interface, inference: LocalInference, **kwargs):
        """Initialize an agent."""
        from agent.memory.hippocampus import Hippocampus

        self.inference = inference
        self.state = AgentState(interface=interface, **kwargs)
        self.hippocampus = Hippocampus(self.state)
        self.set_system_prompt()

    def set_system_prompt(self):
        prompt = self.state.system_prompt
        prompt += "\n\n---- Tools ----\n"
        for tool in self.state.tools_map.values():
            prompt += f"\n---{tool.name}---\n"
            prompt += f"Tool name: {tool.name}\n"
            prompt += f'Tool description: \n"""\n{tool.description}\n"""\n'
            prompt += f"Tool schema: \n{json.dumps(tool.schema, indent=2)}\n"
        prompt += "\n---- End of tools ----\n"

        tool_use_token_start = self.inference.front_end.control_tokens.tool_use_token_start
        tool_use_token_end = self.inference.front_end.control_tokens.tool_use_token_end
        prompt += f'Starting delimiter: "{tool_use_token_start}\\n"\n'
        prompt += f'Ending delimiter: "{tool_use_token_end}".\n'
        prompt += f'Wrap tool calls between the delimiters "{tool_use_token_start}\\n" and "\\n{tool_use_token_end}".\n'
        prompt += "Any other text is yours to use as you see fit and is not shown to the user.\n"
        system_message = Event(role="system", content=prompt)
        self.hippocampus.append_to_history(system_message)

    @property
    def should_act(self) -> bool:
        """
        Whether the agent should act.

        The agent should act if the current step number is less than or equal to the
        maximum number of sub-steps.

        Step count is incremented by 1 after each action, and is reset to 0 when the
        agent returns to human control.
        """
        return self.state.step_number <= MAX_SUB_STEPS

    async def __call__(self) -> None:
        try:
            await self.loop()
        except Exception as e:
            await self.state.interface.exit_program(e)

    async def loop(self) -> None:
        """
        Run the agent within a loop.

        The agent will take action until it reaches the maximum number of sub-steps.
        """
        while self.should_act:
            if self.state.step_number == 0:
                message = await self.state.interface.get_input()
                if isinstance(message, Event):
                    self.hippocampus.append_to_history(message)
                    await self.state.interface.show_output(message)
                elif message:
                    message = Event(role="user", content=str(message))
                    self.hippocampus.append_to_history(message)
                    await self.state.interface.show_output(message)
                elif message is None:
                    break

            await self.run()

        if not self.should_act:
            message = Event(role="system", content="Returning to human control.")
            await self.state.interface.show_output(message)
            self.state.step_number = 0
            self.state.continue_message_id = None
            await self.loop()
        else:
            await self.state.interface.exit_program()

    async def run(self):
        """
        Run the agent.

        The agent's memory is used to create a prompt for a large language model.
        The model can either:
        - Return a message to the user
        - Call a tool
        """
        prompt = [m.to_dict() for m in self.hippocampus.messages]
        tools = [tool for tool in self.state.tools_map.values()]
        tool_calls: dict[str, list[dict[str, Any]]] = {}
        for m in self.hippocampus.messages:
            if m.state == State.TOOL_RESULT or m.state == State.TOOL_ERROR:
                if m.id not in tool_calls:
                    tool_calls[m.id] = [m.to_dict()]
                else:
                    tool_calls[m.id].append(m.to_dict())

        inference_config = {
            "prompt": prompt,
            "tools": tools,
            "tool_calls": tool_calls,
            "add_reminders": False,
            "add_generation_prompt": True,
            "continue_message_id": self.state.continue_message_id,
            "prefill": None,
        }
        model_output = await self.inference(**inference_config)
        for new_message in model_output:
            self.hippocampus.append_to_history(new_message)
            for tool_called in new_message.tool_calls:
                tool_result = self.use_tool(tool_called)
                self.hippocampus.append_to_history(tool_result)
                await self.state.interface.show_output(tool_result)

                if tool_result.state == State.ASSISTANT_RESPONSE:
                    self.state.step_number = 0
                    self.state.continue_message_id = None
                else:
                    self.state.step_number += 1
                    self.state.continue_message_id = new_message.id

    @staticmethod
    async def create(
        interface: Interface, inference: LocalInference | None = None
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

    def use_tool(self, tool_call: ToolCall) -> Event:
        """Handle a single function call, execute the function, and return results.

        Args:
            function_call: The function call to handle.

        Returns:
            A tuple containing:
            - The message with the function call result.
            - A boolean indicating if the function call failed.
        """

        try:
            tool = self.state.tools_map[tool_call.function.name]
            with self.state.interface.console.status(f"[bold yellow]Calling {tool.name}..."):
                result = tool(self, **tool_call.function.arguments)
            if isinstance(result, Event):
                result.id = tool_call.id
                return result
            return Event(
                role="ipython",
                content=str(result),
                name=tool_call.function.name,
                event_id=tool_call.id,
                state=State.TOOL_RESULT,
            )
        except Exception as e:
            return Event(
                role="ipython",
                content=str(e),
                name=tool_call.function.name,
                event_id=tool_call.id,
                state=State.TOOL_ERROR,
            )

    def __repr__(self) -> str:
        return f"Agent({self.state})"
