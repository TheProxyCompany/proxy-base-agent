import sys
from io import StringIO

from agent.agent import Agent
from agent.interaction import Interaction


def execute_code(self: Agent, code: str | list[str]) -> Interaction:
    """
    Execute Python code in the agent's environment. Only the Python standard library is available.
    No imports are permitted. STDOUT is captured and returned as a string to the agent.

    It should be used for:
    - String manipulation and counting
    - Math and Arithmetic
    - Logic validation
    - Miscellaneous simple tasks

    Aim for one-liners; simple code is easier to parse and execute.
    Avoid unnecessary variables; keep code blocks focused and concise.

    Args:
        code (str | list[str]):
            Python code to execute. Must be valid Python 3.x syntax.
            Simple scripts only.
    """

    # Capture stdout
    stdout = StringIO()
    sys.stdout = stdout

    try:
        # Execute the code and capture any return value
        lines_of_code: str = code if isinstance(code, str) else "\n".join(code)
        compiled_code = compile(lines_of_code.strip(), "<string>", "exec")
        exec_locals = {}
        exec(compiled_code, globals(), exec_locals)
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
        sys.stdout = sys.__stdout__
        breakpoint()

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
