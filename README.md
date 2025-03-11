# Proxy Base Agent

A stateful, tool-enabled agent built with the Proxy Structuring Engine.

## Overview

The Proxy Base Agent uses the [Proxy Structuring Engine](https://github.com/TheProxyCompany/proxy-structuring-engine) to implement a state machine architecture that guides language models through planning and action phases.

## Architecture

```
                    ┌───────────────────┐
                    │                   │
                    ▼                   │
        ┌──────────────────────────────────────────────┐
        │                   PLAN                       │ ◀─ loops (min=x, max=y)
        │ ┌─────────┐  ┌──────────┐  ┌───────────────┐ │
        │ │THINKING │  │SCRATCHPAD│  │INNER MONOLOGUE│ │
        │ └─────────┘  └──────────┘  └───────────────┘ │
        └────────────────────┬─────────────────────────┘
                             │
                             ▼
            ┌───────────────────────────────┐
            │           TAKE ACTION         │
            │ ┌─────────┐        ┌────────┐ │
            │ │  TOOLS  │        │ PYTHON │ │
            │ └─────────┘        └────────┘ │
            └───────────────────────────────┘
                              │
                              ▼
                        ┌─────────┐
                        │  DONE   │
                        └─────────┘
```

The agent transitions between two main phases:

1. **Planning states** (Thinking, Scratchpad, Inner Monologue)
2. **Action states** (Tool calls, Python code)

## Installation

Requires Python 3.11+

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

- [Proxy Structuring Engine (PSE)](https://github.com/TheProxyCompany/proxy-structuring-engine)
- [MLX Proxy](https://github.com/TheProxyCompany/mlx-proxy)
