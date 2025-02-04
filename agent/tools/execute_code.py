import sys
from io import StringIO

from agent.agent import Agent
from agent.interaction import Interaction


def execute_code(self: Agent, code: str) -> Interaction:
    """
    This tool can be used to execute arbitrary python code.
    The code is executed in the same environment as the agent.
    Use for character level string manipulation, complex arithmetic,
    logic validation, and other tasks that are not easily expressed
    in natural language.

    Use sparingly. Only the standard library is available, no imports.

    Arguments:
        code (str): The python code to execute.
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
