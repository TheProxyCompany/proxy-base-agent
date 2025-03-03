from collections.abc import Sequence
from typing import Any

from pse.types.base.any import AnyStateMachine
from pse.types.base.encapsulated import EncapsulatedStateMachine
from pse.types.base.loop import LoopStateMachine
from pse.types.grammar import BashStateMachine, PythonStateMachine
from pse.types.json import json_schema_state_machine
from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse_core import StateGraph
from pse_core.state_machine import StateMachine

DEFAULT_DELIMITERS = {
    "thinking": ("```thinking\n", "\n```"),
    "scratchpad": ("```scratchpad\n", "\n```"),
    "tool": ("```json\n", "\n```"),
    "bash": BashStateMachine.delimiters or ("```bash\n", "\n```"),
    "python": PythonStateMachine.delimiters or ("```python\n", "\n```"),
}


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
        tools: Sequence[dict[str, Any]] | None = None,
        use_python: bool = False,
        use_bash: bool = False,
        force_planning: bool = True,
        max_planning_loops: int = 3,
        thinking_delimiters: tuple[str, str] | None = None,
        scratchpad_delimiters: tuple[str, str] | None = None,
        tool_delimiters: tuple[str, str] | None = None,
        bash_delimiters: tuple[str, str] | None = None,
        python_delimiters: tuple[str, str] | None = None,
    ) -> None:
        self.delimiters = {
            "thinking": thinking_delimiters or DEFAULT_DELIMITERS["thinking"],
            "scratchpad": scratchpad_delimiters or DEFAULT_DELIMITERS["scratchpad"],
            "tool": tool_delimiters or DEFAULT_DELIMITERS["tool"],
            "bash": bash_delimiters or DEFAULT_DELIMITERS["bash"],
            "python": python_delimiters or DEFAULT_DELIMITERS["python"],
        }

        super().__init__(
            state_graph=self.build_state_graph(
                tools=tools,
                use_python=use_python,
                use_bash=use_bash,
                force_planning=force_planning,
                max_planning_loops=max_planning_loops,
                delimiters=self.delimiters,
            ),
            start_state="plan",
            end_states=["done"],
        )

    def build_state_graph(
        self,
        tools: Sequence[dict[str, Any]] | None,
        use_python: bool,
        use_bash: bool,
        force_planning: bool,
        max_planning_loops: int,
        delimiters: dict[str, tuple[str, str]],
    ) -> StateGraph:
        """Construct the agent's state graph."""

        state_graph: StateGraph = {
            "plan": [
                (
                    LoopStateMachine(
                        AnyStateMachine(
                            [
                                FencedFreeformStateMachine(
                                    "thinking",
                                    delimiters["thinking"],
                                    char_min=50,
                                    char_max=1000,
                                ),
                                FencedFreeformStateMachine(
                                    "scratchpad",
                                    delimiters["scratchpad"],
                                    char_min=50,
                                    char_max=1000,
                                ),
                            ]
                        ),
                        min_loop_count=int(force_planning),
                        max_loop_count=max_planning_loops,
                    ),
                    "take_action",
                )
            ],
            "take_action": [],
        }

        if tools:
            _, tools_sm = json_schema_state_machine(
                tools, delimiters=delimiters["tool"]
            )
            state_graph["take_action"].append((tools_sm, "done"))

        if use_python:
            state_graph["take_action"].append(
                (
                    EncapsulatedStateMachine(
                        state_machine=PythonStateMachine,
                        delimiters=delimiters["python"],
                    ),
                    "done",
                )
            )

        if use_bash:
            state_graph["take_action"].append(
                (
                    EncapsulatedStateMachine(
                        state_machine=BashStateMachine,
                        delimiters=delimiters["bash"],
                    ),
                    "done",
                )
            )

        return state_graph

    def thinking_explanation(self) -> str:
        explanation = f"""
        The thinking state represents the agent's internal thoughts and plans.
        The agent should use this like a human would think aloud.
        You should wrap the thinking in {self.delimiters["thinking"][0]!r} and {self.delimiters["thinking"][1]!r} tags.
        """
        return explanation

    def scratchpad_explanation(self) -> str:
        explanation = f"""
        The scratchpad state represents a "scratchpad" where the agent can scribble down ideas, thoughts, and plans.
        The agent should use this like a human might use a physical scratchpad with pen and paper.
        You should wrap the scratchpad in {self.delimiters["scratchpad"][0]!r} and {self.delimiters["scratchpad"][1]!r} tags.
        """
        return explanation

    def tool_explanation(self) -> str:
        explanation = f"""
        The tool state represents a tool call.
        You should wrap the tool call in {self.delimiters["tool"][0]!r} and {self.delimiters["tool"][1]!r} tags.
        """
        return explanation

    def bash_explanation(self) -> str:
        explanation = f"""
        The bash state represents a a bash terminal, where the agent can run commands.
        You should wrap the bash command in {self.delimiters["bash"][0]!r} and {self.delimiters["bash"][1]!r} tags.
        The agent should use this like a human would use a bash terminal.
        Do not do anything with the bash terminal that you would not do with a real bash terminal.
        """
        return explanation

    def python_explanation(self) -> str:
        explanation = f"""
        The python state represents a python interpreter, where the agent can run python code.
        You should wrap the python code in {self.delimiters["python"][0]!r} and {self.delimiters["python"][1]!r} tags.
        The agent should use this like a human would use a python interpreter.
        No imports are available, and assume Python 3.10+ syntax.
        """
        return explanation

    @property
    def state_prompt(self) -> str:
        explanation = f"""
        The agent follows a sequence of steps.
        First, the agent plans. During planning, it can "think" or use a "scratchpad."
        "Thinking" is like thinking quietly to itself. The "scratchpad" is like writing notes.
        The agent can switch back and forth between thinking and using the scratchpad.
        After planning, the agent acts. It can use a tool, write Python code, or use bash commands (if available).

        The agent has different states it can be in:

        - Thinking:
        {self.thinking_explanation()}

        - Scratchpad:
        {self.scratchpad_explanation()}

        - Tool Use:
        {self.tool_explanation()}

        - Python Code:
        {self.python_explanation()}

        - Bash Commands:
        {self.bash_explanation()}

        The agent moves between these modes depending on what it outputs.
        Make sure to always use the correct tags around your output to show which mode you are in.
        """
        return explanation
