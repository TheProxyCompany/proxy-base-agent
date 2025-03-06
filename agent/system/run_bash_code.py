import subprocess

from agent.agent import Agent
from agent.system.interaction import Interaction

DEFAULT_TIMEOUT_SECONDS = 30

async def run_bash_code(
    self: Agent,
    code: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Interaction:
    try:
        result = subprocess.run(
            ["bash", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        output = f"Execution timed out after {timeout_seconds} seconds"

    return Interaction(
        role=Interaction.Role.TOOL,
        content=output,
        title=f"{self.name}'s bash",
        color="cyan",
        emoji="terminal",
    )
