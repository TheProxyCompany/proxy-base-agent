import logging
import os
import sys

import questionary
from rich.align import Align
from rich.emoji import Emoji
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

    async def get_input(self, prompt: str | None = None) -> object | None:
        """Gets user input from the command line using `questionary`.

        Returns:
            object: The user input, or `None` if the user enters Ctrl+C
                 (KeyboardInterrupt).
        """
        user_input: str | None = await questionary.text(
            message=prompt or "Enter your message [enter to send, Ctrl+C to exit]:",
            qmark=">",
        ).ask_async()
        # Clear the line after input
        sys.stdout.write("\033[1A\033[2K\033[G")
        sys.stdout.flush()

        if not user_input:
            return user_input

        return Event(
            content=user_input.strip(),
            state=EventState.USER,
            name="User",
        )

    async def show_output(self, input: object | list[object]) -> None:
        """Handles and displays different types of messages.

        Args:
            message: The `Message` object to be handled.
        """
        if isinstance(input, list):
            for message in input:
                await self.show_output(message)
            return

        if not isinstance(input, Event):
            return

        if input.image_path:
            await self.render_image(input.image_path)
            return

    async def show_live_output(self, output: object) -> None:
        """Show partial output."""
        from pse.structuring_engine import EngineOutput

        if not isinstance(output, EngineOutput):
            return

        renderables = []

        if output.buffer:
            print(output.buffer)
            buffer_markdown_content = Markdown(
                str(output.buffer),
                justify="left",
                code_theme="monokai",
                inline_code_lexer="text",
                inline_code_theme="solarized-dark",
            )
            renderables.append(buffer_markdown_content)

        if output.value:
            print(output.value)
            value_markdown_content = Markdown(
                str(output.value),
                justify="left",
                code_theme="monokai",
                inline_code_lexer="json",
                inline_code_theme="solarized-dark",
            )
            renderables.append(value_markdown_content)

        # if renderables:
        #     panel = Panel(
        #         Columns(renderables),
        #         title="Output",
        #         title_align="left",
        #         expand=True,
        #     )

        #     # self.console.print(panel)

        #     self.live = Live(
        #         panel,
        #         console=self.console,
        #         vertical_overflow="visible",
        #         transient=True,
        #     )

    async def _display_message(
        self,
        message: Event,
        title: str,
        border_style: str,
        emoji: str | None = None,
    ) -> None:
        """Displays a message with consistent formatting and an optional emoji.

        Args:
            message: The `Message` object to be displayed.
            title: The title of the panel.
            border_style: The style of the panel border (Rich library color).
            emoji (Optional[str]): The name of the emoji to include in the
                                    panel title (uses Rich library emojis).
                                    Defaults to `None`.
        """
        title_text = f"{Emoji(emoji)} {title}" if emoji else title
        text = (
            f"{message.content}\n\n💭 *{message.inner_thoughts}*"
            if message.inner_thoughts
            else message.content
        )
        self.console.print(
            Align.left(
                Panel(
                    Markdown(str(text), justify="left"),
                    title=title_text,
                    title_align="left",
                    border_style=border_style,
                    expand=self.PANEL_EXPAND,
                    width=self.PANEL_WIDTH,
                )
            )
        )
        self.console.print()

    async def show_error_message(
        self, message: Event | None = None, e: Exception | None = None
    ) -> None:
        """Display an error message with a warning emoji."""
        if not message or e:
            return

        error_message = message or Event(role="system", content=f"{e}")
        await self._display_message(error_message, "Error", "red", "warning")

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
        except Exception as error:
            await self.show_error_message(e=error)

    async def exit_program(self, error: Exception | None = None) -> None:
        """Exits the program with a goodbye message and a waving hand emoji."""
        self.console.print(f"{Emoji('wave')} [bold]Goodbye![/bold]")

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
            self.live = None
            self.live_content = ""
            self.console.print()
