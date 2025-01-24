import os

import requests

from agent.agent import Agent
from agent.message import Message, MessageState
from agent.prompts import load_prompt_template


def web_search(self: Agent, query: str, max_results: int = 5) -> Message:
    """
    Perform a web search and read a summary of the results.
    This function allows you to search the web for information based on a given query.

    Args:
        self (Agent): The agent instance.
        query (str): The search query string. Text only. An artificial assistant will be created to perform the search.
        max_results (Optional[int]): The maximum number of search results to return. Defaults to 5.

    Returns:
        Message: A message containing the search results or an error message if the search fails.
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return Message(
            role="ipython",
            content="Web search not available. API key not set",
            state=MessageState.TOOL_ERROR,
            name=self.state.name + "'s web search",
        )

    try:
        template = load_prompt_template("web_search")
        system_instructions = template.format(agent_name=self.state.name)
    except Exception as e:
        return Message(
            role="ipython",
            content="You couldn't find the prompt to use the web search tool. The error was: "
            + str(e),
            state=MessageState.TOOL_ERROR,
            name=self.state.name + "'s web search",
        )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    breakpoint()
    payload = {
        "max_results": max_results,
        # "model": self.state.interface.config.brain.web_search_model,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": query},
        ],
    }

    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()
        search_results = response.json()

        if not search_results or "choices" not in search_results:
            raise ValueError("No valid choices in search response")

        choices = search_results["choices"]
        formatted_results = ""
        for choice in choices:
            if "message" in choice and "content" in choice["message"]:
                formatted_results += "\n" + str(choice["message"]["content"])

        return Message(
            role="ipython",
            content=formatted_results,
            state=MessageState.TOOL_RESULT,
            name=self.state.name + "'s web search result",
        )
    except Exception as e:
        error_message = f"Failed to perform web search: {e}"
        return Message(
            role="ipython",
            content=error_message,
            state=MessageState.TOOL_ERROR,
            name=self.state.name + "'s web search",
        )
