import logging
import os
import sys
import traceback

import questionary
from rich.console import RenderableType
from rich.emoji import Emoji
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from agent.event import Event, EventState
from interface import Interface

logger = logging.getLogger(__name__)


class CLIInterface(Interface):
    """Command-line interface for interacting with the Brain agent.

    This class implements the AgentInterface and provides methods for
    displaying different message types with consistent formatting
    using Rich library elements (Panels, Markdown, Emojis).
    """

    PANEL_WIDTH = 120
    PANEL_EXPAND = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._renderable: RenderableType | None = None
        self.live = Live(
            console=self.console,
            vertical_overflow="visible",
            get_renderable=lambda: self._renderable or Markdown(""),
        )
        self.structured_output = None
        self.buffer = None

    async def get_input(self, prompt: str | None = None) -> object | None:
        """Gets user input from the command line using `questionary`.

        Returns:
            object: The user input, or `None` if the user enters Ctrl+C
                 (KeyboardInterrupt).
        """
        exit_phrases = ["exit", "quit", "q", "quit()", "exit()"]
        user_input: str | None = await questionary.text(
            message=prompt or "Enter your message [enter to send, Ctrl+C to exit]:",
            qmark=">",
        ).ask_async()
        # Clear the line after input
        sys.stdout.write("\033[1A\033[2K\033[G")
        sys.stdout.flush()

        if user_input is None:
            return user_input
        elif user_input.lower() in exit_phrases:
            return None

        return Event(
            content=user_input.strip(),
            state=EventState.USER,
        )

    async def show_output(self, input: object | list[object]) -> None:
        """Handles and displays different types of messages.

        Args:
            message: The `Message` object to be handled.
        """

        if not isinstance(input, Event):
            breakpoint()
            return

        if input.image_path:
            await self.render_image(input.image_path)
            return

        content = input.content

        if not content:
            return
        style = input.styling
        emoji = Emoji(style["emoji"])
        panel_style = {
            "border_style": style["color"],
            "title": f"{emoji} {input.name or style['title']}",
            "title_align": "left",
            "expand": False,
            "width": self.PANEL_WIDTH,
        }

        if subtitle := input.buffer:
            panel_style["subtitle"] = subtitle
            panel_style["subtitle_align"] = "left"

        self.console.print(
            Panel(
                Markdown(
                    str(content or "\n\n"),
                    justify="left",
                    code_theme="monokai",
                    inline_code_lexer="text",
                    inline_code_theme="solarized-dark",
                ),
                **panel_style,
            )
        )

    async def show_live_output(self, output: object) -> None:
        """Show partial output."""

        if not isinstance(output, tuple):
            raise ValueError("Output is not a tuple")

        if output[1]:
            self.structured_output = Markdown(
                str(output[1]),
                justify="left",
                code_theme="monokai",
                inline_code_lexer="json",
                inline_code_theme="solarized-dark",
            )
            # if we're switching from buffer to structured output, print the buffer
            if self._renderable == self.buffer:
                self.console.print(self.buffer)

            self._renderable = self.structured_output
        elif output[0]:
            self.buffer = Markdown(
                str(output[0]),
                justify="left",
                code_theme="monokai",
                inline_code_lexer="text",
                inline_code_theme="solarized-dark",
            )
            self._renderable = self.buffer
        self.live.start()

    async def show_error_message(
        self, message: Event | None = None, e: Exception | None = None
    ) -> None:
        """Display an error message with a warning emoji."""
        if not message or e:
            return

        error_message = message or Event(role="system", content=f"{e}")
        await self.show_output(error_message)

    async def render_image(self, image_url: str) -> None:
        """Displays an image from a URL, with optional caption and thoughts.

        Args:
            image_url: The URL of the image to be displayed.
        """
        from urllib.request import urlopen

        from imgcat import imgcat
        from PIL import Image

        try:
            img = Image.open(urlopen(image_url))
            imgcat(img)
            breakpoint()
        except Exception as error:
            await self.show_error_message(e=error)

    async def exit_program(self, error: Exception | None = None) -> None:
        """
        Exits the program.
        """
        if error:
            title = f"{Emoji(name='exclamation')} Error"
            content = f"```pytb\n{traceback.format_exc()}\n```"
            border_style = "red"
        else:
            title = f"{Emoji('wave')} Goodbye"
            content = "*Program terminated.*"
            border_style = "blue"

        markdown = Markdown(
            content,
            justify="left",
            inline_code_theme="solarized-dark",
        )

        panel = Panel(
            markdown,
            title=title,
            title_align="left",
            border_style=border_style,
            expand=True,
        )
        self.console.print(panel)

    async def clear(self) -> None:
        """
        Clears the terminal screen based on the operating system.
        """
        if sys.platform.startswith("win"):
            os.system("cls")  # For Windows
        else:
            os.system("clear")  # For Unix/Linux/macOS

    async def end_live_output(self) -> None:
        """
        End the live output.
        """
        if self.live:
            self.live.stop()
            self.console.print()
