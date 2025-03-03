import asyncio
import logging
import os
import sys

from agent.agent import Agent
from agent.interface.cli_interface import CLIInterface

# Set up logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[\033[1;33m%(levelname)s\033[0m] \033[34m%(message)s\033[0m",
    stream=sys.stdout,
)

agent_kwargs = {
    "max_tokens": 1000,
    "temp": 1.0,
    "min_p": 0.02,
    "min_tokens_to_keep": 10,
    "add_generation_prompt": True,
    "prefill": "",
    "seed": 11,
    "include_python": True,
    "include_bash": True,
}

async def main():
    interface = CLIInterface()
    try:
        agent = await Agent.create(interface, **agent_kwargs)
        await agent.loop()
    except Exception as error:
        await interface.exit_program(error)

asyncio.run(main())
