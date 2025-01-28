import os

import requests

from agent.agent import Agent
from agent.interaction import Interaction
from agent.inference.prompts import load_prompt_template


def web_search(self: Agent, query: str, max_results: int = 1) -> Interaction:
    """
    Perform a web search and read an AI generated summary of the results.

    Args:
        query (str): The search query string.
        max_results (int): The maximum number of search results to return. Defaults to 1.
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    template = load_prompt_template("web_search")
    if not template:
        raise ValueError("No prompt template found for web search")
    system_instructions = template.format(agent_name=self.name)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "max_results": max_results,
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": query},
        ],
    }

    response = requests.post(
        "https://api.perplexity.ai/chat/completions", headers=headers, json=payload
    )
    response.raise_for_status()
    search_results = response.json()
    if "choices" not in search_results:
        raise ValueError("Search results not found")

    choices = search_results["choices"]
    for choice in choices:
        if "message" in choice and "content" in choice["message"]:
            return Interaction(
                content=choice["message"]["content"],
                role=Interaction.Role.TOOL,
                name=self.name + "'s web search result",
                buffer=query,
                color="yellow",
                emoji="magnifying_glass",
            )

    return Interaction(
        content="No search results found",
        role=Interaction.Role.TOOL,
        name=self.name + "'s web search result",
        buffer=query,
        color="red",
        emoji="magnifying_glass",
    )
