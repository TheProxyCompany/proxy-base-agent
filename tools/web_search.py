import os

import requests

from agent.agent import Agent, AgentStatus
from agent.event import Event, EventState
from agent.prompts import load_prompt_template


def web_search(self: Agent, query: str, max_results: int = 5) -> Event:
    """
    Perform a web search and read a summary of the results.
    This function allows you to search the web for information based on a given query.

    Args:
        self (Agent): The agent instance.
        query (str): The search query string. Text only. An artificial assistant will be created to perform the search.
        max_results (int): The maximum number of search results to return. Defaults to 5.

    Returns:
        Message: A message containing the search results or an error message if the search fails.
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    template = load_prompt_template("web_search")
    system_instructions = template.format(agent_name=self.state.name)

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
    formatted_results = ""
    for choice in choices:
        if "message" in choice and "content" in choice["message"]:
            formatted_results += "\n" + str(choice["message"]["content"])
            break

    self.status = AgentStatus.SUCCESS
    return Event(
        content=formatted_results,
        state=EventState.TOOL,
        name=self.state.name + "'s web search result",
    )
