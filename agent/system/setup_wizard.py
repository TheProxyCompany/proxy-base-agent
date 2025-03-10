"""Setup wizard for configuring and initializing an agent."""
from agent.agent import Agent
from agent.interface import Interface
from agent.llm.local import LocalInference

# Default agent configuration
DEFAULT_AGENT_KWARGS = {
    # Generation parameters
    "max_tokens": 5000,
    "temp": 1.0,
    "min_p": 0.02,
    "min_tokens_to_keep": 9,
    "character_max": 2500,
    # Prompt configuration
    "add_generation_prompt": True,
    "prefill": "",
    "seed": 11,
    # Feature toggles
    "include_python": False,
    "include_bash": False,
    "enable_voice": False,
    # Planning behavior
    "max_planning_loops": 5,
    "force_planning": False,
    # Caching options
    "reuse_prompt_cache": True,
    "cache_system_prompt": True,
    # MCP configuration
    "default_mcp_servers": [],
    "connect_default_mcp_servers": True,
    "include_pause_button": True,
}

async def get_boolean_option(
    interface: Interface,
    option_name: str,
    default: bool = False,
) -> bool:
    """Get a boolean option from the user."""
    formatted_option = option_name.replace("_", " ").title()
    response = await interface.get_input(
        message=formatted_option,
        choices=["Yes", "No"],
        default="Yes" if default else "No",
    )
    value = response.content if hasattr(response, "content") else response
    return value == "Yes" or (isinstance(value, str) and value.lower() == "yes")

async def get_numeric_option(
    interface: Interface,
    option_name: str,
    default: float,
    min_value: float = 0.0,
    max_value: float | None = None,
    float_type: bool = False,
):
    """Get a numeric option from the user with range information."""
    formatted_option = option_name.replace("_", " ").title()
    # Add range to the option name for clarity
    if max_value is not None:
        formatted_option = f"{formatted_option} ({min_value}-{max_value})"

    response = await interface.get_input(
        message=formatted_option,
        default=str(default),
    )

    value: str = response.content
    try:
        # Convert string to appropriate numeric type
        converted_value = float(value) if float_type else int(value)

        # Apply bounds checking
        if min_value is not None and converted_value < min_value:
            return min_value
        if max_value is not None and converted_value > max_value:
            return max_value
        return converted_value
    except ValueError:
        return default

async def show_section_header(interface: Interface, title: str):
    """Display a formatted section header in the interface."""
    # Clean formatting that will work with the CLI interface
    await interface.show_output(f"\n--- {title} ---\n")

async def configure_basic_options(interface: Interface) -> tuple[str, str, str, str]:
    """Configure basic agent identity and model options."""
    # Get agent identity
    agent_name = await Agent.get_agent_name(interface)
    system_prompt_name = await Agent.get_agent_prompt(interface)

    # Get model information
    model_path = await Agent.get_model_path(interface)
    frontend_response = await interface.get_input(
        message="Inference Backend",
        choices=["mlx", "torch"],
        default="mlx",
    )
    chosen_frontend: str = frontend_response.content

    return agent_name, system_prompt_name, model_path, chosen_frontend

async def setup_agent(interface: Interface) -> Agent:
    """
    Run an interactive setup wizard to configure and initialize an agent.

    This wizard groups configuration options into logical sections and guides
    the user through the setup process step by step.

    Args:
        interface: The interface to use for the setup wizard

    Returns:
        A configured and initialized Agent instance
    """
    await interface.clear()
    await interface.show_output("=== PROXY AGENT CONFIGURATION ===\n")

    # ----- Basic Configuration -----
    await show_section_header(interface, "ESSENTIAL SETTINGS")
    agent_name, system_prompt_name, model_path, chosen_frontend = await configure_basic_options(interface)

    # Start with default configuration
    agent_kwargs = DEFAULT_AGENT_KWARGS.copy()

    # Ask for configuration mode (simple vs. advanced)
    config_mode_response = await interface.get_input(
        message="Configuration Mode",
        choices=["Simple", "Advanced"],
        default="Simple",
    )
    config_mode = config_mode_response.content if hasattr(config_mode_response, "content") else config_mode_response

    if config_mode == "Simple":
        # Simple configuration - no capabilities options in simple mode
        pass
    else:
        # Advanced - go through all configuration sections

        # ----- Capabilities -----
        await show_section_header(interface, "CAPABILITIES")

        # Code execution capabilities
        agent_kwargs["include_python"] = await get_boolean_option(
            interface,
            "Python execution",
            DEFAULT_AGENT_KWARGS["include_python"]
        )
        agent_kwargs["include_bash"] = await get_boolean_option(
            interface,
            "Bash execution",
            DEFAULT_AGENT_KWARGS["include_bash"]
        )

        # Voice capabilities
        agent_kwargs["enable_voice"] = await get_boolean_option(
            interface,
            "Enable voice synthesis",
            DEFAULT_AGENT_KWARGS["enable_voice"]
        )

        # ----- Agent Behavior -----
        await show_section_header(interface, "PLANNING BEHAVIOR")
        agent_kwargs["force_planning"] = await get_boolean_option(
            interface,
            "Force planning phase",
            DEFAULT_AGENT_KWARGS["force_planning"]
        )
        agent_kwargs["max_planning_loops"] = await get_numeric_option(
            interface,
            "maximum planning loops",
            DEFAULT_AGENT_KWARGS["max_planning_loops"],
            min_value=1,
            max_value=10
        )

        # ----- Performance Options -----
        await show_section_header(interface, "PERFORMANCE")
        agent_kwargs["reuse_prompt_cache"] = await get_boolean_option(
            interface,
            "Reuse prompt cache",
            DEFAULT_AGENT_KWARGS["reuse_prompt_cache"]
        )
        agent_kwargs["cache_system_prompt"] = await get_boolean_option(
            interface,
            "Cache system prompt",
            DEFAULT_AGENT_KWARGS["cache_system_prompt"]
        )

        # ----- Inference Parameters -----
        if await get_boolean_option(interface, "Configure inference parameters", False):
            await show_section_header(interface, "INFERENCE PARAMETERS")
            agent_kwargs["temp"] = await get_numeric_option(
                interface,
                "temperature",
                DEFAULT_AGENT_KWARGS["temp"],
                min_value=0.0,
                max_value=2.0,
                float_type=True,
            )
            agent_kwargs["min_p"] = await get_numeric_option(
                interface,
                "minimum p",
                DEFAULT_AGENT_KWARGS["min_p"],
                min_value=0.0,
                max_value=1.0,
                float_type=True,
            )
            agent_kwargs["max_tokens"] = await get_numeric_option(
                interface,
                "maximum tokens",
                DEFAULT_AGENT_KWARGS["max_tokens"],
                min_value=100,
                max_value=16000,
            )
            agent_kwargs["character_max"] = await get_numeric_option(
                interface,
                "maximum characters per state",
                DEFAULT_AGENT_KWARGS["character_max"],
                min_value=100,
                max_value=10000,
            )

        # ----- MCP Configuration -----
        if await get_boolean_option(interface, "Configure MCP servers", False):
            await show_section_header(interface, "MCP SERVER CONFIGURATION")
            agent_kwargs["connect_default_mcp_servers"] = await get_boolean_option(
                interface,
                "Connect to default MCP servers",
                DEFAULT_AGENT_KWARGS["connect_default_mcp_servers"]
            )

        # ----- Interface Configuration -----
        await show_section_header(interface, "INTERFACE CONFIGURATION")
        agent_kwargs["include_pause_button"] = await get_boolean_option(
            interface,
            "Include pause button (spacebar)",
            DEFAULT_AGENT_KWARGS["include_pause_button"]
        )

        # ----- Reproducibility -----
        if await get_boolean_option(interface, "Configure random seed", False):
            await show_section_header(interface, "REPRODUCIBILITY")
            use_random_seed = await get_boolean_option(
                interface,
                "Use random seed",
                True
            )
            if not use_random_seed:
                agent_kwargs["seed"] = await get_numeric_option(
                    interface,
                    "fixed random seed",
                    DEFAULT_AGENT_KWARGS["seed"],
                    min_value=0,
                    max_value=1000000
                )

    # ----- Initialization -----
    await show_section_header(interface, "INITIALIZATION")

    # Initialize agent and model
    with interface.console.status("Loading model and initializing agent..."):
        # Load the model
        inference = LocalInference(model_path, frontend=chosen_frontend)

        # Create the agent
        agent = Agent(
            agent_name,
            system_prompt_name,
            interface,
            inference,
            **agent_kwargs,
        )

    if agent_kwargs["connect_default_mcp_servers"]:
        with interface.console.status("Connecting to default MCP servers..."):
            # get the default mcp servers
            default_mcp_servers = agent_kwargs["default_mcp_servers"]
            for server in default_mcp_servers:
                # connect to the server
                new_tools = await agent.mcp_host.connect_to_server(server)
                # add the tools to the agent
                agent.add_tools(new_tools)

            if len(default_mcp_servers) > 0:
                # reconfigure with the new tools
                agent.configure(True)

    return agent
