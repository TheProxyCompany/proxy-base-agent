import asyncio

from agent.agent import Agent
from agent.interaction import Interaction


def ask_system(self: Agent, question: str) -> Interaction:
    """
    Ask a question to the system.
    Do not use this tool to ask the user anything.

    Arguments:
        question (str): The question.
    """
    answer = asyncio.run(self.interface.get_input(message=question))
    if isinstance(answer, Interaction):
        return answer
    else:
        return Interaction(content=answer, role=Interaction.Role.SYSTEM)
