import asyncio
import json
import logging
import time
import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from agent.agent import Agent
from agent.interface import Interface
from agent.llm.local import LocalInference
from agent.state import AgentState
from agent.system.interaction import Interaction
from agent.system.memory import Memory

# Placeholder for configuration - replace with your config loading
AGENT_NAME = "API_Agent"
SYSTEM_PROMPT_NAME = "base"  # Default prompt name
MODEL_PATH = "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"  # Default model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --- Minimal Interface for API Context ---
class APIBridgeInterface(Interface):
    """A minimal Interface implementation for the API.
    Most methods are no-ops as I/O happens via HTTP req/res.
    """

    def __init__(self):
        # No console needed for API usually
        pass

    async def get_input(self, **kwargs) -> Interaction:
        # Input comes from the HTTP request, not solicited here
        raise NotImplementedError("APIBridgeInterface does not solicit input.")

    async def show_output(self, output: object | list[object]) -> None:
        # Output goes back in the HTTP response
        logger.debug(f"[APIBridgeInterface] Suppressed output: {output}")
        pass

    def show_live_output(self, state: AgentState | None, output: object) -> None:
        # Live output handled by streaming response if enabled
        logger.debug(f"[APIBridgeInterface] Suppressed live output: {state} - {output}")
        pass

    def end_live_output(self) -> None:
        pass

    async def render_image(self, image_url: str) -> None:
        logger.warning(
            "[APIBridgeInterface] Image rendering not supported in API response."
        )
        pass

    async def exit_program(self, error: Exception | None = None) -> None:
        logger.info("[APIBridgeInterface] Exit called (no-op in API context).")
        pass

    async def clear(self) -> None:
        pass  # No console to clear


# --- Globals ---
agent_instance: Agent | None = None


# --- Context Manager for FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs on startup
    global agent_instance
    logger.info("Initializing Agent for API...")
    try:
        api_interface = APIBridgeInterface()
        # TODO: Replace hardcoded config with dynamic loading
        inference = LocalInference(model_path=MODEL_PATH)
        agent_instance = Agent(
            name=AGENT_NAME,
            system_prompt_name=SYSTEM_PROMPT_NAME,
            interface=api_interface,
            inference=inference,
            include_pause_button=False,
        )
        logger.info(f"Agent '{AGENT_NAME}' initialized successfully.")
    except Exception as e:
        logger.error(f"FATAL: Error initializing agent: {e}\n{traceback.format_exc()}")
        agent_instance = None
        # Decide if you want the app to start without the agent
        # raise RuntimeError("Failed to initialize Agent") from e

    yield  # API is running

    # Runs on shutdown
    if (
        agent_instance
        and hasattr(agent_instance, "mcp_host")
        and hasattr(agent_instance.mcp_host, "cleanup")
    ):
        logger.info("Cleaning up agent MCP host resources...")
        try:
            await agent_instance.mcp_host.cleanup()
            logger.info("Agent MCP host cleanup complete.")
        except Exception as e:
            logger.error(f"Error during agent MCP host cleanup: {e}")
    agent_instance = None
    logger.info("Agent resources cleaned up.")


# --- Helper Functions ---
# Removed initialize_agent and cleanup_agent, using lifespan now


# Corrected Role Mapping Functions
def map_openai_role_to_interaction(role: str) -> Interaction.Role:
    """Maps OpenAI role string to Interaction.Role enum."""
    role_map = {
        "user": Interaction.Role.USER,
        "assistant": Interaction.Role.ASSISTANT,
        "system": Interaction.Role.SYSTEM,
        "tool": Interaction.Role.TOOL,
        # "function": InteractionRole.TOOL, # Alias if needed
    }
    try:
        return role_map[role.lower()]
    except KeyError:
        logger.warning(f"Unknown OpenAI role '{role}', defaulting to USER.")
        return Interaction.Role.USER


def map_interaction_role_to_openai(role: Interaction.Role) -> str:
    """Maps Interaction.Role enum to OpenAI role string."""
    # Assumes InteractionRole enum members have a .name (like USER, ASSISTANT)
    return role.name.lower()


# --- Pydantic Models for Request Validation ---
class OpenAIChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = (
        None  # Allow null content for tool calls
    )
    name: str | None = None
    tool_calls: list[dict] | None = None  # For assistant messages
    tool_call_id: str | None = None  # For tool messages


class ChatCompletionRequest(BaseModel):
    model: str  # Although we might ignore it if there's only one agent config
    messages: list[OpenAIChatMessage]
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    max_tokens: int | None = None
    stream: bool | None = False
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    # Add other OpenAI parameters as needed


# --- FastAPI Application ---
app = FastAPI(
    title="Proxy Base Agent - OpenAI Compatible API",
    description="Provides an OpenAI-compatible `/chat/completions` endpoint.",
    version="0.1.0",
    lifespan=lifespan,  # Use lifespan context manager
)


# --- Core Endpoint ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handles OpenAI-compatible chat completion requests."""
    if agent_instance is None:
        raise HTTPException(
            status_code=503, detail="Agent not initialized or initialization failed."
        )

    start_time = time.time()
    original_memory = agent_instance.memory  # Store original memory
    request_memory = Memory()  # Create temporary memory for this request

    try:
        # 1. Prepare temporary memory for this request
        # Add system prompt first
        request_memory.append_to_history(agent_instance.system_prompt)

        history_interactions: list[Interaction] = []
        for msg in request.messages:
            interaction_role = map_openai_role_to_interaction(msg.role)
            content = None
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                content = json.dumps(msg.content)

            interaction_kwargs = {}
            if msg.tool_calls:
                interaction_kwargs["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                interaction_kwargs["tool_call_id"] = msg.tool_call_id

            # Add interaction to the *list* first
            interaction = Interaction(
                role=interaction_role, content=content, **interaction_kwargs
            )
            history_interactions.append(interaction)
            # Add to the temporary memory object
            request_memory.append_to_history(interaction)

        # Temporarily replace agent's memory with the request-specific one
        agent_instance.memory = request_memory

        # 2. Trigger agent processing for one turn using the temporary memory
        # Ensure agent status allows generation
        agent_instance.step_number = 0  # Reset step count
        agent_instance.status = Agent.Status.PROCESSING

        await agent_instance.generate_action()  # Runs inference based on temp memory
        # take_action modifies the *current* agent_instance.memory (which is request_memory)
        await agent_instance.take_action()

        # 3. Retrieve the latest assistant response from the temporary memory
        # Iterate backwards through the temporary memory events dict
        final_assistant_response = None
        if hasattr(request_memory, "events") and isinstance(
            request_memory.events, dict
        ):
            for interaction in reversed(request_memory.events.values()):
                if interaction.role == Interaction.Role.ASSISTANT:
                    final_assistant_response = interaction
                    break
        else:
            logger.error(
                "Could not access temporary memory events to find assistant response."
            )

        if not final_assistant_response:
            logger.error(
                "Agent did not produce an assistant response in temporary memory."
            )
            raise HTTPException(
                status_code=500, detail="Agent failed to generate a response."
            )

        agent_response = final_assistant_response

    except Exception as e:  # Catch broad exceptions during processing
        logger.error(f"Error during agent processing: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail="Internal server error during agent processing."
        ) from e
    finally:
        # CRITICAL: Restore the original memory object regardless of success/failure
        agent_instance.memory = original_memory
        logger.debug("Restored original agent memory.")
    # --- End Agent Interaction ---

    end_time = time.time()
    logger.info(f"Agent processing time: {end_time - start_time} seconds")
    # 4. Format response
    response_id = f"chatcmpl-{int(time.time() * 1000)}"  # Pseudo-random ID
    created_time = int(time.time())
    model_name = request.model  # Echo back the requested model

    if request.stream:
        # Use SSE for streaming
        return StreamingResponse(
            stream_generator(agent_response, response_id, created_time, model_name),
            media_type="text/event-stream",
        )
    else:
        # Return a standard JSON response
        return JSONResponse(
            content=interaction_to_openai_response(
                agent_response, response_id, created_time, model_name
            ),
            status_code=200,
        )


async def stream_generator(
    agent_response: Interaction, response_id: str, created_time: int, model_name: str
) -> AsyncGenerator[str, None]:
    """Generates Server-Sent Events (SSE) for streaming responses.
    NOTE: Currently SIMULATES streaming based on the final response.
    True streaming requires agent.generate_action to yield chunks.
    """
    # 1. Send the initial chunk (role)
    # Use Interaction.Role for mapping
    initial_chunk_delta = {"role": map_interaction_role_to_openai(agent_response.role)}
    # TODO: Handle initial tool_calls delta for streaming if applicable
    # Example structure (needs verification against OpenAI spec):
    # if hasattr(agent_response, 'tool_calls') and agent_response.tool_calls:
    #    initial_chunk_delta["tool_calls"] = [
    #        {"index": i, "id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": ""}}
    #        for i, tc in enumerate(getattr(agent_response, 'tool_calls', []))
    #    ]

    initial_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_name,
        "choices": [{"index": 0, "delta": initial_chunk_delta, "finish_reason": None}],
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"
    await asyncio.sleep(0.01)  # Small delay

    # 2. Send content chunks (simulated)
    content = agent_response.content or ""
    # TODO: Handle streaming arguments for tool calls
    # If tool calls exist, content might be None initially, then arguments stream.
    # Requires adapting the agent_response structure or tool call streaming logic.

    chunk_size = 10  # Simulate chunking speed
    for i in range(0, len(content), chunk_size):
        chunk_content = content[i : i + chunk_size]
        delta = {"content": chunk_content}
        chunk_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"
        await asyncio.sleep(0.05)  # Simulate generation time

    # 3. Send the final chunk with finish reason
    finish_reason = "stop"  # Default
    if hasattr(agent_response, "tool_calls") and agent_response.tool_calls:
        # Check if the agent response object actually has tool_calls attribute populated
        finish_reason = "tool_calls"
    # TODO: Add other finish reasons like 'length' if applicable

    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"

    # 4. Send the [DONE] marker
    yield "data: [DONE]\n\n"


def interaction_to_openai_response(
    agent_response: Interaction, response_id: str, created_time: int, model_name: str
) -> dict[str, Any]:
    """Formats a final agent Interaction into an OpenAI JSON response."""
    message_dict: dict[str, Any] = {
        # Use Interaction.Role for mapping
        "role": map_interaction_role_to_openai(agent_response.role),
        "content": agent_response.content or "",  # Ensure content is not None
    }

    # Add tool_calls if present in the agent's final interaction
    # Ensure the structure matches OpenAI's expected format
    if hasattr(agent_response, "tool_calls") and agent_response.tool_calls:
        # Assuming agent_response.tool_calls is already in the correct list-of-dicts format
        # Example expected OpenAI format for a tool call:
        # { "id": "call_abc123", "type": "function", "function": {"name": "my_func", "arguments": "{\n  \"arg1\": \"value1\"\n}"}}
        message_dict["tool_calls"] = agent_response.tool_calls
        message_dict["content"] = None  # Content is null when tool_calls are present

    finish_reason = "stop"
    if hasattr(agent_response, "tool_calls") and agent_response.tool_calls:
        finish_reason = "tool_calls"
    # TODO: Add other finish reasons

    choice = {
        "index": 0,
        "message": message_dict,
        "finish_reason": finish_reason,
    }

    # TODO: Implement token counting if possible/needed
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model_name,
        "choices": [choice],
        "usage": usage,
    }


# --- Uvicorn Runner (for direct execution) ---
if __name__ == "__main__":
    import uvicorn

    print("Starting API server via Uvicorn...")
    # Make sure agent initialization runs before starting server if run this way
    # asyncio.run(initialize_agent()) # Running this here can cause issues with FastAPI startup events
    uvicorn.run(
        "agent.interface.api_interface.serving:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
    )  # Use reload for dev
