# The Proxy Base Agent

<p align="center">
  <strong>A Stateful, Tool-Enabled Agent Powered by the Proxy Structuring Engine</strong>
</p>

## Overview

The Proxy Base Agent showcases a powerful, structured implementation built on top of the [Proxy Structuring Engine (PSE)](https://github.com/TheProxyCompany/proxy-structuring-engine).

The Proxy Base Agent leverages a state machine architecture to enable reliable, deterministic agent behavior while maintaining the creative power of large language models.

It's designed for both performance and flexibility, allowing you to create agents that can:

- Execute complex, multi-step tasks
- Utilize external tools and APIs
- Maintain context and state across interactions

## Installation

The Proxy Base Agent requires Python 3.11 or newer.

```bash
# Clone the repository
git clone https://github.com/TheProxyCompany/agent.git
cd agent

# Install using uv
uv pip install -e .
```

## Usage

To run the Proxy Base Agent:

```bash
# Start the agent with the interactive setup wizard
python -m agent
```

## Related Projects

- [Proxy Structuring Engine (PSE)](https://github.com/TheProxyCompany/proxy-structuring-engine) - The core technology powering the Proxy Base Agent
- [MLX Proxy](https://github.com/TheProxyCompany/mlx-proxy) - Optimized MLX implementations used by this agent
