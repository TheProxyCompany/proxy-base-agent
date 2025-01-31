import os

import requests

from agent.agent import Agent
from agent.interaction import Interaction

WEB_SEARCH_TEMPLATE = """
You are an advanced AI system with the capability to perform web searches to gather real-time information.

Your task is to find accurate and relevant information based on the given query.

You are acting on behalf of {agent_name}, an AI agent.
Your goal is to assist {agent_name} by performing web searches and providing relevant information.

Your final output should be a well-structured and informative response based on the search results.
The response should be formatted as a clear and concise summary, addressing the original query comprehensively.
Ensure it is concise and to the point, and do not include any additional information that is not relevant to the query.
Respond with a short, bulleted list of the most relevant information. 5 bullets max.
"""

def web_search(self: Agent, query: str, max_results: int = 1) -> Interaction:
    """
    Perform a web search and get a generated summary of the results.
    Only perform a web search to answer questions that require real-time information. Expensive.

    Args:
        query (str): The search query string.
        max_results (int): The maximum number of search results to return. Defaults to 1.
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "max_results": max_results,
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {
                "role": "system",
                "content": WEB_SEARCH_TEMPLATE.format(agent_name=self.name),
            },
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
                title=self.name + "'s web search result",
                subtitle=query,
                color="dim white",
                emoji="magnifying_glass",
            )

    return Interaction(
        content="No search results found",
        role=Interaction.Role.TOOL,
        title=self.name + "'s web search result",
        subtitle=query,
        color="red",
        emoji="magnifying_glass",
    )
