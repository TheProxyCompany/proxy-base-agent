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
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_script:
        temp_script.write(code.encode())
        temp_script.flush()

        try:
            result = subprocess.run(
                [sys.executable, temp_script.name],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=True,  # Let us process non-zero exits as needed
            )
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            output = f"Execution timed out after {timeout_seconds} seconds"
        finally:
            os.remove(temp_script.name)

    return Interaction(
        role=Interaction.Role.TOOL,
        content=output,
        title=f"{self.name}'s code",
        color="cyan",
        emoji="computer",
    )
