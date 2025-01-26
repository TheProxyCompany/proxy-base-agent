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

async def main():
    interface = CLIInterface()
    try:
        agent = await Agent.create(interface)
        await agent()
    except Exception as error:
        await interface.exit_program(error)
        import traceback
        traceback.print_exc()
        breakpoint()

asyncio.run(main())
