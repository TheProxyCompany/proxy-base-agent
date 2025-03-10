MCP_PROMPT = """
    Model Context Protocol (MCP) is a protocol that allows agents to connect to external services.
    As an agent, treat MCP servers as extensions of your capabilities that you use to access real-world data, systems, and tools.

    Using the MCP:
    - First, assess what capabilities you need to accomplish the task
    - Use `list_mcp_servers` to explore available specialized servers
    - Fuzzy select which server is most likely to have the capabilities you need
        - example: if you need access to real-time weather data, but there is no specific weather server, choose the web search server
    - Use `add_mcp_server` with the precise identifier from `list_mcp_servers` to connect to the appropriate server
    - After connecting, the server's tools are integrated into your standard toolkit and you can use them as if they were part of your core capabilities

    MCP servers are pre-vetted extensions of your capabilities, so no user permission is needed.
    Automatically connect to an MCP server "behind the scenes" when you determine their tools would be beneficial to the task at hand.

    You may be connected to multiple MCP servers simultaneously.
    Avoid explicitly mentioning your use of multiple MCP servers unless directly asked or specifically instructed to switch servers.
    You will not be able to use a server that requires credentials unless the user provides the necessary credentials to their .env file.

    Be purposeful and goal-oriented when using MCP servers.
    Only connect to an MCP server if you need to use its tools to accomplish the task at hand.
    Do not connect to an MCP server just because it is available - only connect when you need to use its tools.
"""
