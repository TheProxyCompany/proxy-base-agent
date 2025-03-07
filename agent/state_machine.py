from pse.types.base.any import AnyStateMachine
from pse.types.base.loop import LoopStateMachine
from pse_core.state_machine import StateMachine

from agent.state import (
    AgentState,
    Bash,
    InnerMonologue,
    Python,
    Reasoning,
    Scratchpad,
    Thinking,
    ToolCallState,
)
from agent.tools import Tool


class AgentStateMachine(StateMachine):
    """
        State machine orchestrating the agent's planning and action phases.

                        ┌───────────────────┐
                        │                   │
                        ▼                   │
            ┌─────────────────────────────────────┐
            │                 PLAN                │ ◀─ loops (min=0, max=3)
            │ ┌─────────┐  ┌──────────┐           │
            │ │THINKING │  │SCRATCHPAD│           │
            │ └─────────┘  └──────────┘           │
            │ ┌─────────┐  ┌───────────────┐      │
            │ │REASONING│  │INNER MONOLOGUE│      │
            │ └─────────┘  └───────────────┘      │
            └──────────────────┬──────────────────┘
                               │
                               ▼
                ┌───────────────────────────────┐
                │           TAKE ACTION         │
                │ ┌─────────┐ ┌────────┐ ┌─────┐│
                │ │  TOOLS  │ │ PYTHON │ │BASH ││
                │ └────┬────┘ └───┬────┘ └──┬──┘│
                └──────┼──────────┼─────────┼───┘
                       │          │         │
                       └──────────┼─────────┘
                                  ▼
                            ┌─────────┐
                            │  DONE   │
                            └─────────┘

    Explanation:
    - The agent begins in PLAN, iteratively cycling (0 to 3 loops) through the unordered states: THINKING, SCRATCHPAD, REASONING, and INNER MONOLOGUE.
    - After planning, it transitions into TAKE ACTION, selecting among TOOLS, PYTHON, or BASH (if enabled).
    - Upon completing the action phase, the agent transitions into DONE.
    """

    def __init__(
        self,
        tools: list[Tool] | None = None,
        use_python: bool = False,
        use_bash: bool = False,
        force_planning: bool = True,
        max_planning_loops: int = 3,
        delimiters_kwargs: dict[str, tuple[str, str] | None] | None = None,
        character_max: int | None = None,
    ) -> None:
        self.states: dict[str, AgentState] = {}
        delimiters = delimiters_kwargs or {}
        planning_states = self.create_planning_states(character_max=character_max, **delimiters)
        action_states = self.create_action_states(
            tools=tools, use_python=use_python, use_bash=use_bash, **delimiters
        )

        super().__init__(
            {
                "plan": [
                    (
                        LoopStateMachine(
                            AnyStateMachine(planning_states),
                            min_loop_count=int(force_planning),
                            max_loop_count=max_planning_loops,
                            whitespace_seperator=True,
                        ),
                        "take_action",
                    )
                ],
                "take_action": [(action, "done") for action in action_states],
            },
            start_state="plan",
            end_states=["done"],
        )

    def create_planning_states(self, character_max=None, **delimiters: tuple[str, str] | None) -> list[StateMachine]:
        thinking_state = Thinking(delimiters.get("thinking"), character_max=character_max)
        self.states[thinking_state.identifier] = thinking_state

        scratchpad_state = Scratchpad(delimiters.get("scratchpad"), character_max=character_max)
        self.states[scratchpad_state.identifier] = scratchpad_state

        inner_monologue_state = InnerMonologue(delimiters.get("inner_monologue"), character_max=character_max)
        self.states[inner_monologue_state.identifier] = inner_monologue_state

        reasoning_state = Reasoning(delimiters.get("reasoning"), character_max=character_max)
        self.states[reasoning_state.identifier] = reasoning_state

        return [
            thinking_state.state_machine,
            scratchpad_state.state_machine,
            inner_monologue_state.state_machine,
            reasoning_state.state_machine,
        ]

    def create_action_states(
        self,
        tools: list[Tool] | None = None,
        use_python: bool = False,
        use_bash: bool = False,
        **delimiters: tuple[str, str] | None,
    ) -> list[StateMachine]:

        action_states = []
        if tools:
            tool_state = ToolCallState(
                tools,
                delimiters.get("tool_call"),
                delimiters.get("tool_list"),
            )
            self.states[tool_state.identifier] = tool_state
            action_states.append(tool_state.state_machine)
        if use_python:
            python_state = Python()
            self.states[python_state.identifier] = python_state
            action_states.append(python_state.state_machine)

        if use_bash:
            bash_state = Bash()
            self.states[bash_state.identifier] = bash_state
            action_states.append(bash_state.state_machine)

        return action_states

    @property
    def prompt(self) -> str:
        explanation = f"""
An agentic system operates through a sequence of states to interact with its environment.

State Transitions:
- Move between states using delimiters to indicate the start and end of a state.
- Each transition should be purposeful and advance toward an underlying goal
- Do not be overly verbose or repetitive

Available States:
{ "\n".join(str(state) for state in self.states.values()) }
Encapsulate all outputs with the correct delimiters corresponding to your current state.
When operating in any state, embody the state's intended purpose rather than verbally confirming your state.
        """
        return explanation
