import asyncio
import logging
import os
import sys

from agent.agent import Agent
from agent.interface.cli_interface import CLIInterface
from agent.llm.local import LocalInference

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
    "min_tokens_to_keep": 9,
    "add_generation_prompt": True,
    "prefill": "",
    "seed": 11,
    "include_python": False,
    "include_bash": False,
    "max_planning_loops": 5,
    "force_planning": False,
}

async def main():
    interface = CLIInterface()
    await interface.clear()

    agent_name = await Agent.get_agent_name(interface)
    system_prompt_name = await Agent.get_agent_prompt(interface)

    model_path = await Agent.get_model_path(interface)
    with interface.console.status("[yellow]Loading model..."):
        inference = LocalInference(model_path, frontend="mlx")
    try:
        agent = Agent(
            agent_name,
            system_prompt_name,
            interface,
            inference,
            **agent_kwargs,
        )
        await agent.loop()
    except Exception as error:
        await interface.exit_program(error)

asyncio.run(main())
