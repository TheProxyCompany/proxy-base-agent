"""Agent module"""

from __future__ import annotations

import atexit
import logging
import uuid
from enum import Enum
from random import randint

from pynput import keyboard as pynput_keyboard

from agent.interface import Interface
from agent.llm import get_available_models
from agent.llm.local import LocalInference
from agent.llm.prompts import get_available_prompts, load_prompt
from agent.mcp import MCP_PROMPT
from agent.mcp.host import MCPHost
from agent.state import AgentState
from agent.state_machine import AgentStateMachine
from agent.system.interaction import Interaction
from agent.system.memory import Memory
from agent.system.voice import VoiceBox
from agent.tools import Tool, ToolCall

logger = logging.getLogger(__name__)

MAX_SUB_STEPS: int = 20

class Agent:
    """
    The main Agent class that orchestrates language model interactions, tool usage, and state transitions.

    This class implements a stateful agent that processes user inputs, generates responses using
    language models, and can interact with the environment through specialized tools. The agent
    follows a state machine architecture that structures its behavior into planning states
    (thinking, inner monologue, etc.) and action states (tool usage, code execution, etc.).

    The agent maintains its conversation history in memory and can be configured with
    different capabilities, such as Python code execution and various
    specialized tools. It also supports features like pause/resume during processing.
    """

    class Status(Enum):
        """
        Enumeration of possible agent status states.

        These states track the operational status of the agent through its lifecycle
        and are used for control flow and UI feedback.
        """
        # Core System States
        PAUSED = "paused"          # Agent execution is temporarily suspended
        PROCESSING = "processing"  # Agent is actively processing
        STANDBY = "standby"        # Agent is ready but not actively processing
        IDLE = "idle"              # Agent is initialized but not engaged

        # Error and Recovery States
        SUCCESS = "success"       # Action completed successfully
        FAILED = "failed"         # Action failed to complete
        RETRYING = "retrying"     # Attempting recovery from failure
        BLOCKED = "blocked"       # Unable to proceed, waiting for external resolution

        def __str__(self):
            return self.value

        @classmethod
        def from_string(cls, state_string: str):
            """Convert a string representation to a Status enum value."""
            return cls(state_string)

    state_machine: AgentStateMachine
    available_states: dict[str, AgentState]
    tools: dict[str, Tool]

    def __init__(
        self,
        name: str,
        system_prompt_name: str,
        interface: Interface,
        inference: LocalInference,
        seed: int | None = None,
        tools: list[Tool] | list[str] | None = None,
        python_interpreter: bool = False,
        max_planning_loops: int = 3,
        force_planning: bool = True,
        character_max: int | None = None,
        include_pause_button: bool = True,
        **inference_kwargs,
    ):
        """
        Initialize an Agent instance with the specified configuration.

        Args:
            name: Human-readable name for this agent instance
            system_prompt_name: Name of the system prompt template to use
            interface: Interface implementation for user interaction
            inference: Inference engine for language model interaction
            seed: Random seed for reproducible generation (default: random)
            tools: List of Tool instances or tool names to load (default: all available)
            python_interpreter: Whether to enable Python code execution (default: False)
            max_planning_loops: Maximum iterations through planning states (default: 3)
            force_planning: Whether to require at least one planning iteration (default: True)
            character_max: Maximum character limit for each state output (default: None)
            include_pause_button: Enable spacebar to pause execution (default: True)
            **inference_kwargs: Additional parameters for the inference engine
        """

        self.seed = seed or randint(0, 1000000)
        self.name = name
        self.status = Agent.Status.IDLE
        self.step_number = 0
        self.prefill = None

        self.system_prompt_name = system_prompt_name
        self.inference_kwargs = inference_kwargs
        self.inference_kwargs["seed"] = self.seed
        self.python_interpreter = python_interpreter
        self.max_planning_loops = max_planning_loops
        self.force_planning = force_planning
        self.character_max = character_max

        self.inference = inference
        self.interface = interface

        self.tools: dict[str, Tool] = {}
        for tool in tools or Tool.load():
            if isinstance(tool, Tool):
                self.tools[tool.name] = tool
            elif isinstance(tool, str):
                self.tools[tool] = Tool.load(file_name=tool)[0]

        self.memory = Memory()
        self.enable_voice = inference_kwargs.pop("enable_voice", False)
        self.voicebox = VoiceBox() if self.enable_voice and VoiceBox.is_downloaded() else None
        self.mcp_host = MCPHost()
        self.configure(set_system_prompt=True)

        if include_pause_button:
            # Set up keyboard listener for pause/resume functionality
            self.keyboard_listener: pynput_keyboard.Listener | None = None
            self._setup_keyboard_listener()

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

    @property
    def system_prompt(self) -> Interaction:
        prompt = load_prompt(self.system_prompt_name)
        if prompt is None:
            raise ValueError(f"No System Prompt Found for {self.system_prompt_name}")

        try:
            formatted_prompt = prompt.format(
                name=self.name,
                state_prompt=self.state_machine.prompt,
                mcp_prompt=MCP_PROMPT,
            )
            prompt = formatted_prompt
        except Exception as e:
            logger.error(f"Error formatting system prompt: {e}")
            pass

        return Interaction(role=Interaction.Role.SYSTEM, content=prompt)

    async def loop(self) -> None:
        """
        Run the agent in a continuous interaction loop.

        This method is the main entry point for agent execution. It:
        1. Gets user input through the configured interface
        2. Processes the input and determines the appropriate response
        3. Generates actions using the language model until reaching limits
        4. Recursively continues the interaction loop

        The loop stops when the agent reaches MAX_SUB_STEPS, receives a shutdown
        signal, or encounters an unrecoverable error.
        """
        message = await self.interface.get_input(
            message="Enter your message [enter to send, Ctrl+C to exit]:",
            qmark=">",
            clear_line=True,
        )
        if isinstance(message, Interaction):
            self.status = Agent.Status.PROCESSING
            if message.content:
                self.memory.append_to_history(message)
                await self.interface.show_output(message)
        elif message is None:
            self.status = Agent.Status.STANDBY
            await self.interface.exit_program()
            return

        self.step_number = 0
        while self.can_act:
            self.step_number += 1
            self.status = Agent.Status.PROCESSING
            await self.generate_action()

        await self.loop()

    async def generate_action(self) -> None:
        """
        Generate an appropriate action using the language model.

        This method:
        1. Resets the inference engine for a fresh generation
        2. Runs inference on the full conversation history
        3. Shows live output as it's being generated
        4. Handles pause/resume functionality
        5. Processes the final structured output for execution

        The generated output is structured according to the agent's state machine,
        which determines what actions (thinking, tool usage, code execution) to take.
        """
        self.inference.engine.reset()
        for _ in self.inference.run_inference(
            prompt=[e.to_dict() for e in self.memory.events.values()],
            **self.inference_kwargs,
        ):
            if live_output := self.inference.engine.get_live_structured_output():
                current_state = self.available_states.get(live_output[0].lower())
                self.interface.show_live_output(current_state, live_output[1])
            else:
                self.interface.end_live_output()

            if self.status == Agent.Status.PAUSED:
                # useful for debugging
                self.interface.end_live_output()
                breakpoint()
                self.status = Agent.Status.PROCESSING

        self.interface.end_live_output()
        await self.take_action()

    async def take_action(self) -> None:
        """
        Take actions based on an event.

        This method handles the event, appends it to the history, and processes
        any tool calls.
        """
        action = Interaction(
            role=Interaction.Role.ASSISTANT,
            name=self.name,
        )
        for state, output in self.inference.engine.get_stateful_structured_output():
            agent_state = self.available_states.get(state)
            if not agent_state:
                logger.warning(f"Unknown state: {state}")
                continue

            match agent_state.identifier:
                case "scratchpad" | "thinking" | "inner_monologue":
                    action.content += agent_state.format(output.strip()) + "\n"

                case "tool_call":
                    tool_call = ToolCall(**output)
                    interaction = await self.use_tool(tool_call)
                    await self.interface.show_output(interaction)
                    action.metadata["tool_call"] = tool_call.to_dict()
                    action.metadata["tool_result"] = interaction.to_dict()

                case "python":
                    from agent.system.run_python_code import run_python_code

                    interaction = await run_python_code(self, output)
                    await self.interface.show_output(interaction)
                    action.metadata["tool_call"] = agent_state.format(output.strip())
                    action.metadata["tool_result"] = interaction.to_dict()


                case _:
                    raise ValueError(f"Unknown structured output: {output}")

        self.memory.append_to_history(action)

    async def use_tool(self, tool_call: ToolCall) -> Interaction:
        """
        Execute a tool call and return the results as an Interaction.

        This method handles both local tools and tools available through MCP servers.
        It manages all aspects of tool execution including error handling and status
        updates.

        Args:
            tool_call: The structured tool call to execute, including the tool name and arguments

        Returns:
            An Interaction object containing either the successful tool results or
            an error message if the tool execution failed
        """
        try:
            tool = self.tools[tool_call.name]
            with self.interface.console.status(f"[yellow]Using {tool_call.name}"):
                if not tool.mcp_server:
                    result = await tool.call(self, **tool_call.arguments or {})
                    assert isinstance(result, Interaction)
                else:
                    result = await self.mcp_host.use_tool(
                        tool.mcp_server,
                        tool_call,
                    )

            return result
        except Exception as e:
            self.status = Agent.Status.FAILED
            return Interaction(
                role=Interaction.Role.TOOL,
                content=f"Tool call failed: {e}",
            )

    def add_tools(
        self,
        new_tools: list[Tool],
        reset_system_prompt: bool = False,
    ):
        """
        Add new tools to the agent.
        """
        self.tools.update({tool.name: tool for tool in new_tools})
        self.configure(reset_system_prompt)

    def configure(self, set_system_prompt: bool = False):
        self.state_machine = AgentStateMachine(
            tools=list(self.tools.values()),
            use_python=self.python_interpreter,
            max_planning_loops=self.max_planning_loops,
            force_planning=self.force_planning,
            delimiters_kwargs=self.inference.front_end.tokenizer.delimiters,
            character_max=self.character_max,
        )
        self.available_states = self.state_machine.states
        self.inference.engine.configure(self.state_machine)
        if set_system_prompt:
            self.memory.update_system_prompt(self.system_prompt)

    @staticmethod
    async def get_agent_name(interface: Interface) -> str:
        """
        Prompt the user for an agent name.

        Returns:
            str: The chosen agent name or a generated UUID if left blank.
        """
        response = await interface.get_input(
            message="Give your agent a name",
            default="Procksy",
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
            message="System Prompt",
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
        available_models["Get a model from HuggingFace"] = ""
        model_name = await interface.get_input(
            message="Select LLM",
            choices=list(available_models.keys()),
            default=next(iter(available_models.keys())),
        )
        model_path = available_models[model_name.content]
        if not model_path:
            huggingface_model_name = await interface.get_input(
                message="Enter HuggingFace Model ID",
                default="mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
            )
            return huggingface_model_name.content

        return model_path

    def toggle_pause(self):
        """Toggle the agent's pause state when the spacebar is pressed."""
        if self.status != Agent.Status.PAUSED:
            self.status = Agent.Status.PAUSED
        else:
            self.status = Agent.Status.PROCESSING
        logger.info(
            f"Agent {'paused' if self.status == Agent.Status.PAUSED else 'resumed'}"
        )

    def _setup_keyboard_listener(self):
        """Set up the keyboard listener with a weak reference to avoid memory leaks."""
        import weakref

        # Create a weak reference to self
        weak_self = weakref.ref(self)

        # Define a callback that uses the weak reference
        def on_key_press(key):
            self_ref = weak_self()
            if (
                self_ref is not None
                and self_ref.status == Agent.Status.PROCESSING
                and key == pynput_keyboard.Key.space
            ):
                self_ref.toggle_pause()
                logger.info(
                    f"Agent {'paused' if self_ref.status == Agent.Status.PAUSED else 'resumed'}"
                )

        # Create and start the listener
        self.keyboard_listener = pynput_keyboard.Listener(on_press=on_key_press)
        self.keyboard_listener.start()

        # Register cleanup function
        def cleanup():
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                self.keyboard_listener = None

        atexit.register(cleanup)
