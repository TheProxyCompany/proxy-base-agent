import sys
from io import StringIO

from agent.agent import Agent
from agent.interaction import Interaction


def execute_code(self: Agent, lines_of_code: list[str]) -> Interaction:
    """
    Execute Python code in the agent's environment.
    STDOUT is captured and returned as a string to the agent.

    This tool provides a sandboxed Python interpreter for performing operations
    that are cumbersome or impossible to express in natural language.
    It should be used for:
    - String manipulation
    - Complex math
    - Logic validation

    Security & Limitations:
    - Only the Python standard library is available
    - No imports are permitted

    Best Practices:
    - Aim for one-liners; simple code is easier to parse and execute.
    - Avoid unnecessary variables; keep code blocks focused and concise.

    Args:
        lines_of_code (list[str]):
            Python code to execute. Must be valid Python 3.x syntax.
            The code will be executed in the order provided.
    """

    # Capture stdout
    stdout = StringIO()
    sys.stdout = stdout

    try:
        # Execute the code and capture any return value
        code = compile("\n".join(lines_of_code), "<string>", "exec")
        exec_locals = {}
        exec(code, globals(), exec_locals)
        output = stdout.getvalue()

        # If there's a return value in the last expression, include it
        if "_" in exec_locals:
            output += str(exec_locals["_"])

        if output:
            result = f"Executed:\n{lines_of_code}\nOutput:\n{output}"
        else:
            result = f"Executed:\n{lines_of_code}\nOutput:\n*No output produced*"

    except Exception as e:
        result = f"Executed:\n{lines_of_code}\nError:\n{e}"

    finally:
        # Restore stdout
        sys.stdout = sys.__stdout__

    return Interaction(
        role=Interaction.Role.TOOL,
        content=result,
        title=f"{self.name}'s code",
        color="cyan",
        emoji="computer",
    )
