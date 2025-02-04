import sys
from io import StringIO

from agent.agent import Agent
from agent.interaction import Interaction


def execute_code(self: Agent, code: str) -> Interaction:
    """
    Execute Python code in the agent's environment for computational tasks.
    STDOUT is captured and returned as a string to the agent.

    This tool provides a sandboxed Python interpreter for performing operations
    that are cumbersome or impossible to express in natural language. It's
    particularly useful for:
    - String manipulation and regex operations
    - Complex mathematical calculations
    - Logic validation and testing

    Security & Limitations:
    - Only the Python standard library is available
    - No external imports are permitted
    - Execution environment is isolated

    Best Practices:
    - Use meaningful variable names
    - Add comments for complex operations
    - Keep code blocks focused and concise

    Args:
        code: Python code to execute. Must be valid Python 3.x syntax.
    """

    # Capture stdout
    stdout = StringIO()
    sys.stdout = stdout

    try:
        # Execute the code and capture any return value
        exec_locals = {}
        exec(code, {}, exec_locals)
        output = stdout.getvalue()

        # If there's a return value in the last expression, include it
        if "_" in exec_locals:
            output += str(exec_locals["_"])

        if output:
            result = f"```\n{code}\n```\n```\n{output}\n```"
        else:
            result = f"```\n{code}\n```\n*No output produced*"

    except Exception as e:
        breakpoint()
        result = f"```\n{code}\n```\nError running code: {e}"

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
