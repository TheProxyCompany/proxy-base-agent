<p align="center">
  <img src="logo.png" alt="Proxy Base Agent" style="object-fit: contain; max-width: 50%; padding-top: 20px;"/>
</p>

<h2 align="center">
  A stateful agent with 100% reliable tool use.
</h2>

<p align="center" style="font-size: 1.2em; width: 80%; margin: 0 auto; padding-bottom: 20px;">
  Build custom agents on any LLM with guaranteed state consistency and 100% reliable tool use.
  <br>
  Powered by the <a href="https://github.com/TheProxyCompany/proxy-structuring-engine">Proxy Structuring Engine (PSE)</a>.
</p>

<p align="center">
  <a href="https://pypi.org/project/proxy-base-agent/"><img src="https://img.shields.io/pypi/v/proxy-base-agent.svg" alt="PyPI Version"></a>
  <a href="https://docs.theproxycompany.com/pba/"><img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation"></a>
  <a href="https://github.com/TheProxyCompany/proxy-base-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

## The Problem: Agents That Don't Work

Most LLM agents rely on fragile prompt engineering. This leads to unpredictable state management, hallucinated tool arguments, and frequent execution failures – making reliable automation nearly impossible.

**The Proxy Base Agent (PBA) is the engineered solution.** PBA uses the [Proxy Structuring Engine (PSE)](https://github.com/TheProxyCompany/proxy-structuring-engine) – our high-performance Hierarchical State Machine (HSM) engine – to enforce reliable, stateful execution for any LLM.

With the Proxy Base Agent, you define your agent's behavior through a defined state graph, and PSE **guarantees** the LLM adheres to it, step-by-step.

## Key Capabilities

*   **Guaranteed Stateful Execution:** Define agent workflows as explicit HSMs (e.g., Plan ➔ Act). PSE ensures the LLM follows the defined states and transitions precisely.
*   **100% Reliable Tool Use:** Eliminate runtime errors from malformed API calls or hallucinated function arguments. PSE guarantees tool calls match their required schema *during* generation.
*   **Dynamic Runtime Adaptation (MCP):** Connect to external Model Context Protocol (MCP) servers on-the-fly. PBA instantly integrates new tools and capabilities with the same structural guarantees, no restarts needed.
*   **Model & Framework Agnostic:** Run reliable agents locally using your preferred LLMs and backends (MLX, PyTorch supported).
*   **Modular & Extensible:** Build specialized agents by adding custom tools, defining new states, or modifying the core HSM architecture.

## How It Works: Reliability Through Structure

PBA's core is an HSM enforced by PSE at runtime:

1.  **HSM Definition:** Agent logic (states like Thinking, Tool Call) is defined as a `StateMachine`. Each state uses a nested PSE `StateMachine` to enforce its specific output structure (e.g., fenced text, JSON schema).
2.  **PSE Runtime Enforcement:** The `StructuringEngine` ensures the LLM generates only valid state transitions and structurally correct output *within* each state. Tool call arguments are guaranteed to match the required schema.
3.  **Dynamic Updates (MCP):** Connecting to an MCP server rebuilds the relevant parts of the HSM and reconfigures PSE instantly, making new tools reliably available.

**PBA doesn't just *ask* the LLM to be stateful and reliable; it *engineers* it through PSE's runtime HSM governance.**

## Installation

```bash
pip install proxy-base-agent
```
*(See [Installation Docs](https://docs.theproxycompany.com/pba/getting-started/installation/) for full details, development setup, and framework extras)*

## Quickstart

Launch the interactive setup wizard to configure your LLM and run the agent:

```bash
python -m agent
```

## Documentation

We've created detailed technical documentation for the Proxy Base Agent:

*   [Overview](https://docs.theproxycompany.com/pba/)
*   [Quickstart](https://docs.theproxycompany.com/pba/getting-started/quickstart/)
*   [Core Concepts](https://docs.theproxycompany.com/pba/concepts/)
*   [Extending the Agent](https://docs.theproxycompany.com/pba/extending/)
*   [API Reference](https://docs.theproxycompany.com/pba/api/)

## The Proxy Company Services

Leverage our foundational expertise to build complex, mission-critical agentic systems with PBA. We offer:

*   **Custom Agent Development:** Tailored agents for your specific workflows.
*   **Advanced Integration:** Connect PBA reliably to proprietary systems and APIs.
*   **Production Support:** Architecture, performance tuning, and support contracts.

➡️ **[Explore Business Services & Schedule Consultation](https://theproxycompany.com/business)**

## License

Apache 2.0 ([LICENSE](LICENSE)). Depends on `pse` (Apache 2.0).

## Citation

```bibtex
@software{Wind_Proxy_Base_Agent_2025,
  author       = {Wind, Jack},
  title        = {{Proxy Base Agent: Reliable AI Execution via Hierarchical State Machines}},
  version      = {1.0.0},
  date         = {2025-04-15},
  url          = {https://github.com/TheProxyCompany/proxy-base-agent},
  publisher    = {The Proxy Company},
  note         = {Leverages the Proxy Structuring Engine (PSE) for runtime guarantees}
}
```
