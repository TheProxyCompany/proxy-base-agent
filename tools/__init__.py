from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import uuid
from collections.abc import Callable
from typing import Any

from pse import callable_to_json_schema


class Tool:
    """A tool is a piece of code that can be invoked by an agent."""

    def __init__(
        self,
        name: str,
        callable: Callable,
        source_type: str = "python",
    ):
        self.name = name
        self.source_type = source_type
        self.source_code = inspect.getsource(callable)
        self.json_schema = callable_to_json_schema(callable)
        self.callable = callable

    def __call__(self, caller: Any, **kwargs) -> Any:
        """
        Call the tool with the given arguments.

        Args:
            caller (Any): The caller of the tool.
            **kwargs: Additional arguments to pass to the tool.

        Returns:
            Any: The result of the tool call.
        """
        arguments = {
            **kwargs,
            "self": caller,
        }
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

        result = self.callable(**arguments)

        arguments.pop("self", None)
        return result

    @staticmethod
    def load(filepath: str | None = None) -> list[Tool]:
        """
        Load a single Python function from a given file and generate its schema.

        Args:
            filepath (str): Path to the Python file containing the function.

        Returns:
            dict[str, dict]: Dictionary containing the module source, function object,
                and JSON schema for the loaded function.

        Raises:
            ModuleNotFoundError: If the module cannot be loaded from the filepath.
            ValueError: If no function matching the module name is found.
            ImportError: If there are errors importing module dependencies.
        """
        path = filepath or os.path.dirname(__file__)

        if os.path.isdir(path):
            tools = []
            for file in os.listdir(path):
                tools.extend(Tool.load(os.path.join(path, file)))
            return tools
        elif (
            not os.path.isfile(path)
            or not path.endswith(".py")
            or path.startswith("__")
        ):
            return []

        # Extract module name from filepath
        module_name = os.path.splitext(os.path.basename(path))[0]

        # Import the module dynamically
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            raise ModuleNotFoundError(f"Cannot load module from {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the function matching the module name
        function = getattr(module, module_name, None)
        if not inspect.isfunction(function):
            raise ValueError(f"No function named '{module_name}' found in {path}.")

        return [Tool(module_name, function)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema": self.json_schema,
        }

    def get_invocation_schema(self) -> dict[str, Any]:
        tool_name = self.name
        tool_schema = self.json_schema
        tool_parameters = tool_schema.get("parameters", {})
        invocation_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "const", "const": tool_name},
                "arguments": {
                    "type": "object",
                    "properties": tool_parameters.get("properties", {}),
                    "required": tool_parameters.get("required", []),
                },
            },
            "required": ["name", "arguments"],
        }
        return invocation_schema


class FunctionCall:
    def __init__(self, name: str, arguments: dict[str, str]):
        self.name = name
        self.arguments: dict[str, str] = arguments

class ToolCall:
    def __init__(
        self,
        tool_call_type: str,
        function: FunctionCall,
        id: str | None = None,
    ):
        self.tool_call_type = tool_call_type
        self.function = function
        self.id: str = id or str(uuid.uuid4())

    def to_dict(self):
        return {
            "id": str(self.id),
            "type": self.tool_call_type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }

    def __str__(self):
        return json.dumps(self.to_dict())

    def __repr__(self):
        return self.__str__()
