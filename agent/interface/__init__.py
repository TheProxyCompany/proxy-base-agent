from abc import ABC, abstractmethod

from rich.console import Console

from agent.interaction import Interaction
from agent.tools import ToolCall


class Interface(ABC):
    """
    This class provides a foundation for creating interfaces that
    handle various types of interactions and display them in different formats.
    """

    def __init__(self):
        self.console = Console()

    @abstractmethod
    async def get_input(self, **kwargs) -> Interaction:
        """
        Get input.

        Returns:
            Interaction: The input.
        """
        pass

    @abstractmethod
    async def show_output(self, output: object | list[object]) -> None:
        """
        Handle and display any type of output.

        Args:
            output (object): The output to be handled and displayed.
        """
        pass

    @abstractmethod
    async def show_tool_use(self, tool_call: ToolCall) -> None:
        """
        Show a tool call.
        """
        pass

    @abstractmethod
    def show_live_output(self, buffer: object, structured: object) -> None:
        """
        Show partial output.

        Args:
            buffer (object): The buffer to be displayed.
            structured (object): The structured output to be displayed.
        """
        pass

    @abstractmethod
    def end_live_output(self) -> None:
        """End live output."""
        pass

    @abstractmethod
    async def render_image(self, image_url: str) -> None:
        """
        Display an image with optional caption and inner thoughts.

        Args:
            image_url (str): The URL of the image to be displayed.
        """
        pass

    @abstractmethod
    async def exit_program(self, error: Exception | None = None) -> None:
        """Exit the program with a goodbye message."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        Clear the console.
        """
        pass


from .cli_interface import CLIInterface  # noqa: E402

__all__ = ["CLIInterface", "Interface"]
