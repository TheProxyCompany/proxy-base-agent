from pse.types.base.any import AnyStateMachine
from pse.types.base.loop import LoopStateMachine
from pse_core.state_machine import StateMachine

from agent.state import AgentState, Bash, Python, Scratchpad, Thinking, ToolCallState
from agent.tools import Tool


class AgentStateMachine(StateMachine):
    """
    A state machine orchestrating the agent's planning and action phases.

                  ┌─────────────────┐
                  │                 │
                  ▼                 │
    ┌─────────────────────────────┐ │
    │             PLAN            │ │
    │  ┌─────────┐    ┌──────────┐│ │- loops (min=0, max=3)
    │  │THINKING │    │SCRATCHPAD││ │
    │  └─────────┘    └──────────┘│ │
    └──────────────┬──────────────┘ │
                   │                │
                   └────────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │          TAKE_ACTION        │
    │                             │
    │  ┌─────────┐┌───────┐┌─────┐│
    │  │  TOOLS  ││PYTHON ││BASH ││
    │  └────┬────┘└───┬───┘└──┬──┘│
    └───────┼─────────┼───────┼───┘
            │         │       │
            └─────────┼───────┘
                      ▼
                 ┌─────────┐
                 │  DONE   │
                 └─────────┘

    The agent begins in PLAN, cycling through THINKING and SCRATCHPAD states to formulate its approach.
    After planning, it transitions to TAKE_ACTION, selecting among TOOLS, PYTHON, or BASH (if enabled).
    Finally, it transitions to DONE.
    """

    def __init__(
        self,
        tools: list[Tool] | None = None,
        use_python: bool = False,
        use_bash: bool = False,
        force_planning: bool = True,
        max_planning_loops: int = 3,
    ) -> None:
        self.states: dict[str, AgentState] = {}

        thinking_state = Thinking()
        self.states[thinking_state.name] = thinking_state

        scratchpad_state = Scratchpad()
        self.states[scratchpad_state.name] = scratchpad_state

        action_states: list[AgentState] = []
        if tools:
            tool_state = ToolCallState(tools)
            self.states[tool_state.name] = tool_state
            action_states.append(tool_state)

        if use_python:
            python_state = Python()
            self.states[python_state.name] = python_state
            action_states.append(python_state)

        if use_bash:
            bash_state = Bash()
            self.states[bash_state.name] = bash_state
            action_states.append(bash_state)

        super().__init__(
            {
                "plan": [
                    (
                        LoopStateMachine(
                            AnyStateMachine(
                                [
                                    Thinking().state_machine,
                                    Scratchpad().state_machine,
                                ]
                            ),
                            min_loop_count=int(force_planning),
                            max_loop_count=max_planning_loops,
                        ),
                        "take_action",
                    )
                ],
                "take_action": [
                    (action.state_machine, "done") for action in action_states
                ],
            },
            start_state="plan",
            end_states=["done"],
        )

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
