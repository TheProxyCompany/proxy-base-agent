"""Agent module"""

from __future__ import annotations

import logging
import uuid
from random import randint

import questionary
from interface import Interface
from model_inference import ModelInference

from agent.memory.hippocampus import Hippocampus
from agent.message import Message, MessageState
from agent.prompts import get_available_prompts, load_prompt_template
from tools import Tool, ToolCall

logger = logging.getLogger(__name__)

MAX_SUB_STEPS: int = 10


class AgentState:
    """Represents the state of an agent."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        interface: Interface,
        seed: int | None = None,
        tools: list[Tool] | None = None,
    ):
        self.interface = interface
        self.name = name
        self.system_prompt = system_prompt
        self.seed = seed or randint(0, 1000000)
        self.step_number = 0
        self.tools_map: dict[str, Tool] = {}
        for tool in tools or Tool.load():
            self.tools_map[tool.name] = tool


class Agent:
    def __init__(self, interface: Interface, inference: ModelInference, **kwargs):
        """Initialize an agent."""
        self.state = AgentState(interface=interface, **kwargs)
        self.hippocampus = Hippocampus(self.state)
        self.inference = inference

    async def __call__(self) -> None:
        while self.state.step_number <= MAX_SUB_STEPS:

            if self.state.step_number > MAX_SUB_STEPS:
                message = Message(role="system", content="Returning to human control.")
                await self.state.interface.handle_output(message)
                break
            elif self.state.step_number == 0:
                message = await self.state.interface.get_input()
                if isinstance(message, Message):
                    self.hippocampus.append_to_messages(message)
                else:
                    message = Message(role="user", content=str(message))
                    self.hippocampus.append_to_messages(message)

            await self.step()

    async def step(self) :
        """
        Process a single step of an agent's interaction.
        """
        prompt = [m.to_dict() for m in self.hippocampus.messages]
        tools = [tool for tool in self.state.tools_map.values()]
        inference_config = {
            "prompt": prompt,
            "tools": tools,
            "add_reminders": True,
            "add_generation_prompt": True,
            "continue_message_id": None,
            "prefill": None,
        }
        model_output = self.inference(**inference_config)
        for new_message in model_output:
            self.hippocampus.append_to_messages(new_message)
            for tool in new_message.tool_calls:
                tool_result = self.use_tool(tool)
                self.hippocampus.append_to_messages(tool_result)
                if tool_result.state != MessageState.ASSISTANT_RESPONSE:
                    self.state.step_number += 1
                else:
                    self.state.step_number = 0

    @staticmethod
    async def create(interface: Interface, inference: ModelInference | None = None) -> Agent:
        agent_name = await Agent.get_agent_name()
        system_prompt = await Agent.get_agent_prompt()
        if inference is None:
            model_path = await Agent.get_model_path()
            inference = ModelInference(model_path)
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
        available_prompts = list(get_available_prompts())
        prompt_name = await questionary.select(
            message="Select a prompt for the agent (hit enter for default):",
            choices=available_prompts,
            default="base",
        ).ask_async()
        return load_prompt_template(prompt_name)

    @staticmethod
    async def get_model_path() -> str:
        """
        Prompt the user for a model path.

        Returns:
            str: The chosen model path.
        """
        MODEL_PATH = "language_models/Llama-3.1-SuperNova-Lite"
        model_path = await questionary.text("Enter a model path (hit enter for default):", default=MODEL_PATH).ask_async()
        return model_path

    def use_tool(self, tool_call: ToolCall) -> Message:
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
            result = tool(self, **tool_call.function.arguments)
            if isinstance(result, Message):
                result.id = tool_call.id
                return result
            return Message(
                role="ipython",
                content=str(result),
                name=tool_call.function.name,
                id=tool_call.id,
                state=MessageState.TOOL_RESULT,
            )
        except Exception as e:
            self.should_handle_tool_result = True

            return Message(
                role="ipython",
                content=str(e),
                name=tool_call.function.name,
                id=tool_call.id,
                state=MessageState.TOOL_ERROR,
            )
