from abc import ABC, abstractmethod

from rich.console import Console


class Interface(ABC):
    """
    Abstract base class for handling Brain-related events.

    This class provides a foundation for creating interfaces
    that handle various types of messages and display them in different formats.
    """

    def __init__(self):
        self.console = Console()

    @abstractmethod
    async def get_input(self, prompt: str | None = None) -> object | None:
        """
        Get user input.

        Returns:
            str: The user input.
        """
        pass

    @abstractmethod
    async def handle_output(self, output: object | list[object]) -> None:
        """
        Handle and display any type of output.

        Args:
            output (object): The output to be handled and displayed.
        """
        pass

    @abstractmethod
    async def show_error_message(self, message: object | None = None, e: Exception | None = None) -> None:
        """
        Display an error message.

        Args:
            e (Exception): The error message to be displayed.
        """
        pass

    @abstractmethod
    async def show_image(self, image: object) -> None:
        """
        Display an image with optional caption and inner thoughts.

        Args:
            image (object): The image to be displayed.
        """
        pass

    @abstractmethod
    async def exit_program(self) -> None:
        """Exit the program with a goodbye message."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        Clear the console.
        """
        pass
