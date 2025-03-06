# The Proxy Agent Framework

<p align="center">
  <strong>A Stateful, Tool-Enabled Agent Powered by the Proxy Structuring Engine</strong>
</p>

## Overview

The Proxy Agent Framework is a powerful, structured agent implementation built on top of the [Proxy Structuring Engine (PSE)](https://github.com/TheProxyCompany/proxy-structuring-engine). It provides a complete solution for creating and deploying intelligent agents with tool-use capabilities, state awareness, and structured output.

This framework leverages PSE's state machine architecture to enable reliable, deterministic agent behavior while maintaining the creative power of large language models. It's designed for both performance and flexibility, allowing you to create agents that can:

- Execute complex, multi-step tasks
- Utilize external tools and APIs
- Maintain context and state across interactions
- Generate reliably structured responses
- Execute code (Python and Bash) when necessary

## Features

- **State Machine Architecture**: Built on PSE's stateful control system to ensure reliable, deterministic behavior
- **Tool Integration**: Easy-to-use system for adding custom tools and capabilities
- **Python and Bash Execution**: Built-in capability to run code safely
- **Multiple Interface Options**: CLI interface included with extensible design
- **Memory Management**: Hippocampus system for managing agent memory and context
- **Local LLM Support**: Optimized for running with local MLX models
- **Voice Capabilities**: Optional voice integration with text-to-speech and speech-to-text

## Installation

```bash
# Clone the repository
git clone https://github.com/TheProxyCompany/agent.git
cd agent

# Install with development dependencies
uv pip install -e .
```

## Usage

```python
from agent.agent import Agent
from agent.interface.cli_interface import CLIInterface
from agent.llm.local import LocalInference

# Initialize interface and inference
interface = CLIInterface()
inference = LocalInference("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit")

# Create an agent
agent = Agent(
    name="Cerebra",
    system_prompt_name="base",
    interface=interface,
    inference=inference,
    include_python=True,
    include_bash=True
)

# Run the agent loop
await agent.loop()
```

## Architecture

The Agent Framework is built on a hierarchical architecture:

1. **Agent Core**: The central coordination system managing state and tool use
2. **State Machine**: PSE-powered component ensuring structured, controlled behavior
3. **Memory (Hippocampus)**: System for maintaining conversation history and context
4. **Tool Integration**: Extensible system for adding capabilities
5. **Interface Layer**: Flexible interfaces for different deployment scenarios

## Extending with Custom Tools

You can easily extend the agent with custom tools:

```python
from agent.tools import Tool
from agent.system.interaction import Interaction

class MyCustomTool(Tool):
    name = "my_custom_tool"
    description = "Performs a custom action"
    
    def __call__(self, agent, **kwargs):
        # Tool implementation
        result = self.perform_action(**kwargs)
        return Interaction(
            role=Interaction.Role.TOOL,
            content=result
        )
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Related Projects

- [Proxy Structuring Engine (PSE)](https://github.com/TheProxyCompany/proxy-structuring-engine) - The core technology powering state control
- [PSE Core](https://github.com/TheProxyCompany/pse_core) - C++ implementation of PSE core components
- [MLX Proxy](https://github.com/TheProxyCompany/mlx-proxy) - Optimized MLX model implementations used by this agent