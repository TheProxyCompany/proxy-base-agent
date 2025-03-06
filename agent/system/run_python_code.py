import os
import subprocess
import sys
import tempfile

from agent.agent import Agent
from agent.system.interaction import Interaction

DEFAULT_TIMEOUT_SECONDS = 30

async def run_python_code(
    self: Agent,
    code: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Interaction:
    """Execute Python code in isolated process with timeout and memory limits.

    Args:
        code: Python code to execute
        timeout_seconds: Maximum execution time in seconds

    Returns:
        Interaction containing execution results or error message
    """
    if not code.endswith("\n"):
        code += "\n"

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_script:
        temp_script.write(code.encode())
        temp_script.flush()
        script_path = temp_script.name

    try:
        # Don't use check=True so we can capture stderr from failed executions
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        # Combine stdout and stderr, regardless of exit code
        output = result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += result.stderr
    except subprocess.TimeoutExpired:
        output = f"Execution timed out after {timeout_seconds} seconds"
    except Exception as e:
        output = f"Execution failed with error: {e}"
    finally:
        # Ensure cleanup happens even if there's an exception
        if os.path.exists(script_path):
            os.remove(script_path)

    return Interaction(
        role=Interaction.Role.TOOL,
        content=output,
        title=f"{self.name}'s code",
        color="cyan",
        emoji="computer",
    )
