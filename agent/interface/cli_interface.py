import logging
import os
import sys
import traceback
from typing import Any

import questionary
from rich.emoji import Emoji
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from agent.interaction import Interaction
from agent.interface import Interface
from agent.tools import ToolCall

logger = logging.getLogger(__name__)


PANEL_WIDTH = 120
PANEL_EXPAND = False


class CLIInterface(Interface):
    """Command-line interface for interacting with the Brain agent.

    This class implements the AgentInterface and provides methods for
    displaying different message types with consistent formatting
    using Rich library elements (Panels, Markdown, Emojis).
    """

    def __init__(self):
        super().__init__()
        self.live = None

    @staticmethod
    def get_panel(content: Any, **panel_style) -> Panel:
        if isinstance(content, list):
            content = "\n\n".join(content)

        markdown = Markdown(
            str(content),
            justify="left",
            code_theme="monokai",
            inline_code_lexer="markdown",
            inline_code_theme="solarized-dark",
        )

        return Panel(markdown, **panel_style)

    async def get_input(self, **kwargs) -> Interaction:
        """
        Gets user input from the command line.
        """
        exit_phrases = ["exit", "quit", "q", "quit()", "exit()"]
        default: str = kwargs.get("default", "")
        answer: str = default

        if kwargs.get("choices"):
            answer: str = await questionary.select(**kwargs).ask_async()
        else:
            answer: str = await questionary.text(**kwargs).ask_async()

        if answer is None or answer.lower() in exit_phrases:
            await self.exit_program()
            sys.exit(0)

        return Interaction(
            content=answer,
            role=Interaction.Role.USER,
        )

    async def show_output(self, output: object | list[object]) -> None:
        """Handles and displays different types of messages.

        Args:
            message: The `Message` object to be handled.
        """

        if not isinstance(output, Interaction):
            return

        if output.image_url:
            await self.render_image(output.image_url)

        style = output.styling
        try:
            emoji = f"{Emoji(style['emoji'])} "
        except Exception:
            emoji = ""

        panel_style = {
            "border_style": style["color"],
            "title": f"{emoji}{style['title'] or output.title}",
            "title_align": "left",
            "subtitle_align": "left",
            "expand": PANEL_EXPAND,
            "width": PANEL_WIDTH,
        }

        if output.scratchpad:
            panel_style["title"] = f"{Emoji('notebook')} scratchpad"
            panel_style["border_style"] = "dim white"
            self.console.print(self.get_panel(output.scratchpad, **panel_style))

        if (subtitle := output.subtitle):
            panel_style["subtitle"] = subtitle

        if output.content:
            self.console.print(self.get_panel(output.content, **panel_style))

        if output.tool_result and isinstance(output.tool_result, Interaction):
            await self.show_output(output.tool_result)

    async def show_tool_use(self, tool_call: ToolCall) -> None:
        """Show a tool call."""
        markdown = Markdown(
            str(tool_call),
            justify="left",
            code_theme="monokai",
            inline_code_lexer="json",
            inline_code_theme="solarized-dark",
        )
        panel_style = {
            "border_style": "blue",
            "title": f"{Emoji('hammer')} {tool_call.name}",
            "title_align": "left",
            "expand": PANEL_EXPAND,
            "width": PANEL_WIDTH,
        }
        self.console.print(Panel(markdown, **panel_style))

    def show_live_output(self, output: object) -> None:
        """Show partial output."""
        if not self.live:
            self.live = Live(
                console=self.console,
                refresh_per_second=10,
                auto_refresh=True,
                transient=True,
                vertical_overflow="visible",
            )
            self.live.start()

        self.live.update(Markdown(str(output)))

    def end_live_output(self) -> None:
        """End live output."""
        self.console.clear_live()
        if self.live:
            self.live.stop()
            self.live = None

        self.console.print()

    async def show_error_message(
        self,
        message: Interaction | None = None,
        e: Exception | None = None,
    ) -> None:
        """Display an error message with a warning emoji."""
        if not message or e:
            return

        error_message = message or Interaction(role="system", content=f"{e}")
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
            os.environ["TOKENIZERS_PARALLELISM"] = "false"  # weird bug
            img = Image.open(urlopen(image_url))
            imgcat(img)
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
