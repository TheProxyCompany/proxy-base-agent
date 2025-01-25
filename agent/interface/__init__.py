from abc import ABC, abstractmethod

from rich.console import Console
from rich.live import Live


class Interface(ABC):
    """
    Abstract base class for handling Brain-related events.

    This class provides a foundation for creating interfaces
    that handle various types of messages and display them in different formats.
    """

    def __init__(self):
        self.console = Console()
        self.live_content = ""
        self.live: Live | None = None

    @abstractmethod
    async def get_input(self, prompt: str | None = None) -> object | None:
        """
        Get user input.

        Returns:
            str: The user input.
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
    async def show_live_output(self, output: object) -> None:
        """
        Show partial output.
        """
        pass

    @abstractmethod
    async def end_live_output(self) -> None:
        """
        End the live output.
        """
        pass

    @abstractmethod
    async def render_image(self, image: object) -> None:
        """
        Display an image with optional caption and inner thoughts.

        Args:
            image (object): The image to be displayed.
        """
        pass

    @abstractmethod
    async def exit_program(self, e: Exception | None = None) -> None:
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
