"""Agent module"""

from __future__ import annotations

import logging
import uuid
from enum import Enum
from random import randint
from typing import TypeVar

from agent.interaction import Interaction
from agent.interface import CLIInterface, Interface
from agent.llm import get_available_models
from agent.llm.local import LocalInference
from agent.memory import Hippocampus
from agent.prompts import get_available_prompts, load_prompt
from agent.state_machine import AgentStateMachine
from agent.system.voice import VoiceBox
from agent.tools import Tool, ToolCall

logger = logging.getLogger(__name__)

MAX_SUB_STEPS: int = 20

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
        include_python: bool = False,
        include_bash: bool = False,
        **inference_kwargs,
    ):
        """Initialize an agent."""

        self.seed = seed or randint(0, 1000000)
        self.name = name
        self.step_number = 0
        self.status = Agent.Status.IDLE

        self.system_prompt_name = system_prompt_name
        self.inference_kwargs = inference_kwargs
        self.include_python = include_python
        self.prefill = None

        self.tools: dict[str, Tool] = {}
        for tool in tools or Tool.load():
            if isinstance(tool, Tool):
                self.tools[tool.name] = tool
            elif isinstance(tool, str):
                self.tools[tool] = Tool.load(file_name=tool)[0]

        self.inference = inference
        self.interface = interface
        self.state_machine = AgentStateMachine(
            tools=[t.to_dict() for t in self.tools.values()],
            use_python=include_python,
            use_bash=include_bash,
            tool_delimiters=self.inference.front_end.tokenizer.control_tokens.tool_use_delimiters(),
        )
        self.inference.engine.configure(self.state_machine)
        self.hippocampus = Hippocampus(self.system_prompt)
        self.voicebox = VoiceBox()

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
            clear_line=True,
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
            try:
                await self.act()
            except Exception as e:
                logger.error(f"Error taking action: {e}")
                self.status = Agent.Status.FAILED
                await self.interface.show_output(
                    Interaction(
                        role=Interaction.Role.ASSISTANT,
                        name=self.name,
                        content=f"Error: {e}",
                    )
                )
                return

        await self.loop()

    async def act(self) -> None:
        """
        Take actions based on an event.

        This method handles the event, appends it to the history, and processes
        any tool calls.
        """

        for token_ids in self.inference.run_inference(
            prompt=[e.to_dict() for e in self.hippocampus.events.values()],
            **self.inference_kwargs,
        ):
            output = self.inference.front_end.tokenizer.decode([token_ids])
            self.interface.show_live_output("", output)

        self.interface.end_live_output()
        for state, output in self.inference.engine.get_structured_output():
            breakpoint()
            match state:
                case "scratchpad", "thinking":
                    print(output)
                    pass
                case "tool_call":
                    interaction = self.use_tool(output)
                    print(interaction)
                    pass
                case "python":
                    from agent.system.run_python_code import run_python_code
                    interaction = await run_python_code(self, output)
                    print(interaction)
                    pass
                case "bash":
                    from agent.system.run_bash_code import run_bash_code
                    interaction = await run_bash_code(self, output)
                    print(interaction)
                    pass
                case _:
                    raise ValueError(f"Unknown structured output: {output}")

        output = Interaction(
            role=Interaction.Role.ASSISTANT,
            name=self.name,
        )
        self.hippocampus.append_to_history(output)
        await self.interface.show_output(output)

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
            system_prompt,
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
        available_models["Download a model from HuggingFace"] = ""
        model_name = await interface.get_input(
            message="Select a model for the agent:",
            choices=list(available_models.keys()),
            default=next(iter(available_models.keys())),
        )
        model_path = available_models[model_name.content]
        if not model_path:
            huggingface_model_name = await interface.get_input(
                message="Download a model from HuggingFace:",
                default="mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
            )
            return huggingface_model_name.content

        return model_path

    @property
    def system_prompt(self) -> Interaction:
        prompt = load_prompt(self.system_prompt_name)
        if prompt is None:
            raise ValueError(f"No System Prompt Found for {self.system_prompt_name}")

        try:
            formatted_prompt = prompt.format(
                name=self.name,
                tool_list=self.tool_list,
                state_prompt=self.state_machine.state_prompt,
            )
            prompt = formatted_prompt
        except Exception:
            pass

        return Interaction(role=Interaction.Role.SYSTEM, content=prompt)

    @property
    def tool_list(self) -> str:
        """
        List of tools available to the agent.
        """
        return "\n---\n".join([str(tool) for tool in self.tools.values()])

    def __repr__(self) -> str:
        return f"{self.name} ({self.status})"
