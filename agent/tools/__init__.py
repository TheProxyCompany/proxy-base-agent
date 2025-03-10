from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
import os
from collections.abc import Callable
from typing import Any

from mcp.types import Tool as MCPTool
from pse.types.json.schema_sources.from_function import callable_to_schema
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Tool:
    """
    A tool is a callable capability that can be invoked by an agent.
    
    Tools are the primary mechanism through which agents interact with external
    systems, APIs, or perform specialized operations. Each tool has a name,
    description, callable implementation, and structured schema that describes
    its parameters and return values.
    
    Tools can be loaded dynamically from Python files or registered manually.
    They support both synchronous and asynchronous execution patterns and
    can be called either directly or through MCP (Model Control Protocol) servers.
    """

    def __init__(
        self,
        name: str,
        description: str,
        callable: Callable | None = None,
        schema: dict[str, Any] | None = None,
        mcp_server: str | None = None,
    ):
        """
        Initialize a new Tool.
        
        Args:
            name: The unique identifier for this tool
            description: Human-readable description of the tool's purpose and usage
            callable: The function that implements the tool's functionality
            schema: JSON schema describing the tool's parameters (auto-generated from callable if None)
            mcp_server: Optional MCP server identifier if this tool is remote
            
        Note:
            Either callable or schema must be provided. If callable is provided,
            the schema will be automatically generated from its signature and docstring.
        """
        self.name = name
        self.description = description
        self.schema = schema
        self.callable = callable
        self.mcp_server = mcp_server
        self.source_code = None
        
        # If a callable is provided, extract its source code and generate schema
        if callable:
            try:
                self.source_code = inspect.getsource(callable)
                self.schema = callable_to_schema(callable)
            except Exception as e:
                logger.warning(f"Could not extract source code for tool {name}: {e}")
                self.source_code = "# Source code not available"

    async def call(self, caller: Any, **kwargs) -> Any:
        """
        Call the tool with the given arguments, handling both sync and async execution.
        
        This method provides a unified interface for tool invocation, automatically
        handling the differences between synchronous and asynchronous callables.
        It prepares arguments according to the tool's requirements and executes
        the underlying function.
        
        Args:
            caller: The agent instance calling this tool (passed as 'self' to the callable)
            **kwargs: Tool-specific arguments as defined in the tool's schema
            
        Returns:
            The result of the tool execution, which could be any type depending on the tool
            
        Raises:
            Exception: If the tool execution fails for any reason
        """
        if not self.callable:
            return None

        arguments = self._prepare_arguments(caller, **kwargs)

        # Check if the callable is a coroutine function
        if inspect.iscoroutinefunction(self.callable):
            result = await self.callable(**arguments)
        else:
            result = self.callable(**arguments)

        return result

    def _prepare_arguments(self, caller: Any, **kwargs) -> dict:
        """
        Prepare the arguments for the tool call.

        Args:
            caller (Any): The caller of the tool.
            **kwargs: Additional arguments to pass to the tool.

        Returns:
            dict: The prepared arguments.
        """
        arguments = {"self": caller, **kwargs}
        spec = inspect.getfullargspec(self.callable)
        annotations = spec.annotations
        for arg_name in spec.args:
            if arg_name not in arguments:
                if spec.defaults and arg_name in spec.args[-len(spec.defaults) :]:
                    default_index = spec.args[::-1].index(arg_name)
                    arguments[arg_name] = spec.defaults[-1 - default_index]

        for name, arg in arguments.items():
            if isinstance(arg, dict) and name in annotations:
                arguments[name] = annotations[name](**arg)

        return arguments

    @staticmethod
    def from_file(filepath: str) -> Tool | None:
        """
        Load a single Tool from a given Python file.
        
        This method loads a function from a Python file and converts it to a Tool.
        The function name must match the file name (without .py extension).
        
        Args:
            filepath: Absolute or relative path to a Python file containing the tool function
            
        Returns:
            A Tool instance if successfully loaded, None otherwise
            
        Note:
            The file must be a valid .py file and the function's docstring will be used
            as the tool's description.
        """
        # valid .py file
        if (
            not os.path.isfile(filepath)
            or not filepath.endswith(".py")
            or os.path.basename(filepath).startswith("__")
        ):
            return None

        # Extract the module name from file name
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        # Import the module dynamically
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if not spec or not spec.loader:
            logger.error(f"Cannot load module from {filepath}")
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # We expect a function that matches the module name
        function = getattr(module, module_name, None)
        if not inspect.isfunction(function):
            logger.warning(f"No function named '{module_name}' found in {filepath}.")
            return None

        return Tool(module_name, description=function.__doc__ or "", callable=function)

    @staticmethod
    def load(
        filepath: str | None = None,
        file_name: str | list[str] | None = None,
    ) -> list[Tool]:
        """
        Load one or more Tools from files.
        
        This versatile method can load tools from:
          1. A single .py file
          2. A directory of .py files
          3. A directory + specific file(s) (by name, without .py)
          
        If no filepath is given, it defaults to the current directory of this module.
        
        Args:
            filepath: Optional path to file or directory. Defaults to module directory.
            file_name: Optional filename or list of filenames to load (without .py extension).
                      If None, load all .py files in the directory (excluding __*.py).
                      
        Returns:
            A list of successfully loaded Tool instances
            
        Examples:
            # Load all tools in the current directory
            tools = Tool.load()
            
            # Load all tools in a specific directory
            tools = Tool.load("/path/to/tools")
            
            # Load a specific tool
            tools = Tool.load(file_name="calculator")
            
            # Load multiple specific tools
            tools = Tool.load(file_name=["calculator", "web_search"])
        """
        if not filepath:
            # Default to the directory containing this file
            filepath = os.path.dirname(__file__)

        found_tools = []
        # -----------------------------------------------------
        # CASE A: `filepath` is a direct *.py file -> load it
        # -----------------------------------------------------
        if os.path.isfile(filepath):
            # Make sure it is actually a .py file
            if filepath.endswith(".py") and not os.path.basename(filepath).startswith(
                "__"
            ):
                tool = Tool.from_file(filepath)
                if tool:
                    found_tools.append(tool)
                else:
                    logger.error(f"Cannot load tool from {filepath}")

        # -----------------------------------------------------
        # CASE B: `filepath` is a directory
        # -----------------------------------------------------
        elif os.path.isdir(filepath):
            # Normalize `file_name` into a list and load all .py files (except __*.py)
            files_to_process = (
                os.listdir(filepath)
                if file_name is None
                else [
                    f if f.endswith(".py") else f + ".py"
                    for f in ([file_name] if isinstance(file_name, str) else file_name)
                ]
            )

            # Process each file
            for f in files_to_process:
                if f.endswith(".py") and not f.startswith("__"):
                    full_path = os.path.join(filepath, f)
                    tool = Tool.from_file(full_path)
                    if tool:
                        found_tools.append(tool)
                    else:
                        logger.error(f"Cannot load tool from {full_path}")

        return found_tools

    def to_dict(self) -> dict[str, Any]:
        schema = self.schema or {}
        return {
            "type": "object",
            "description": self.description or self.name,
            "properties": {
                "intention": {
                    "type": "string",
                    "description": "Your reason for using this specific tool, and intended outcome.",
                    "minLength": 10,
                },
                "name": {"const": self.name},
                "arguments": schema.get("parameters", schema),
            },
            "required": ["intention", "name", "arguments"],
        }

    def __getattribute__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            schema = object.__getattribute__(self, "schema")
            if name in schema:
                return schema[name]
            return None

    def __str__(self) -> str:
        tool = self.to_dict().get("properties", {})
        tool_str = f'\nTool name: "{self.name}"'
        tool_str += f'\nTool description:\n{self.description}'
        tool_str += f'\nTool schema:\n{json.dumps(tool, indent=2)}'
        return tool_str

    @staticmethod
    def from_mcp_tool(mcp_tool: MCPTool, server_id: str) -> Tool:
        """
        Convert an tool from the MCP protocol to a local Tool object.
        """
        schema = mcp_tool.inputSchema
        return Tool(mcp_tool.name, mcp_tool.description or "", schema=schema, mcp_server=server_id)


class ToolCall(BaseModel):
    """
    A structured representation of a tool invocation.
    
    This class encapsulates all the information needed to invoke a tool,
    including the tool name, arguments, and the agent's intention for
    using the tool. It serves as a bridge between the language model output
    and the actual tool execution.
    
    The BaseModel inheritance provides automatic validation of the structure,
    ensuring that all required fields are present and correctly formatted.
    """

    intention: str
    """The reason or goal of the tool call. Minimum 10 characters."""
    name: str
    """The name of the tool to call."""
    arguments: dict[str, Any] | None = None
    """The arguments to pass to the tool."""

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
