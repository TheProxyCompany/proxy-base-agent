from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import uuid
from collections.abc import Callable
from typing import Any

from pse.util import callable_to_json_schema
from pydantic import BaseModel


class Tool:
    """A tool is a piece of code that can be invoked by an agent."""

    def __init__(
        self,
        name: str,
        callable: Callable,
        source_type: str = "python",
    ):
        self.name = name
        self.callable = callable
        self.source_type = source_type
        self.source_code = inspect.getsource(callable)
        self.json_schema = callable_to_json_schema(callable)
        self.json_schema["schema"] = self.get_invocation_schema()

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
            "self": caller,
            **kwargs,
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
    def load(filepath: str | None = None, file_name: str | None = None) -> list[Tool]:
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

        if file_name:
            path = os.path.join(path, file_name)

        if os.path.isdir(path):
            tools = []
            for file in os.listdir(path):
                new_tools = Tool.load(path, file)
                tools.extend(new_tools)
            return tools
        elif (
            not os.path.isfile(path)
            or not path.endswith(".py")
            or (file_name and file_name.startswith("__"))
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.get_invocation_schema(),
        }

    def __getattribute__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            json_schema = object.__getattribute__(self, "json_schema")
            if name in json_schema:
                return json_schema[name]
            return None

    def __str__(self) -> str:
        return f"{self.name}:\n{self.description}"

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)



class ToolUse(BaseModel):
    tool_id: str
    tool_type: str
    function: FunctionCall

    def __init__(
        self,
        tool_type: str,
        function: FunctionCall | Any,
        tool_id: str | None = None,
    ):
        if function and not isinstance(function, FunctionCall):
            name = function.get("name", "")
            arguments = function.get("arguments", {})
            function = FunctionCall(name=name, arguments=arguments)
        super().__init__(
            tool_id=tool_id or str(uuid.uuid4()),
            tool_type=tool_type,
            function=function,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def __str__(self) -> str:
        return json.dumps(self.to_dict())

    def __repr__(self) -> str:
        return self.__str__()


class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]
