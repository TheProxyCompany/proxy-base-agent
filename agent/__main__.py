import asyncio
import logging
import os
import sys

from agent.interface.cli_interface import CLIInterface
from agent.system.setup_wizard import setup_agent

# Set up logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[\033[1;33m%(levelname)s\033[0m] \033[34m%(message)s\033[0m",
    stream=sys.stdout,
)

async def main():
    """
    Initialize and run the agent with an interactive setup wizard.
    """
    # Create the interface
    interface = CLIInterface()

    try:
        # Run the setup wizard to configure and initialize the agent
        agent = await setup_agent(interface)

        # Start the agent loop
        await agent.loop()
    except Exception as error:
        # Handle any exceptions
        await interface.exit_program(error)

# Run the main function
asyncio.run(main())
