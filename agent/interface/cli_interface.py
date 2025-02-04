import logging
import os
import sys
import traceback

import questionary
from rich.align import Align
from rich.console import Group, RenderableType
from rich.emoji import Emoji
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from agent.interaction import Interaction
from agent.interface import Interface
from agent.tools import ToolCall

logger = logging.getLogger(__name__)


PANEL_WIDTH = 100
PANEL_EXPAND = True


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
    def get_panel(interaction: Interaction, **panel_style) -> RenderableType:
        if isinstance(interaction.content, list):
            interaction.content = "\n\n".join(interaction.content)

        markdown = Markdown(
            str(interaction.content),
            justify="left",
            code_theme="monokai",
            inline_code_lexer="markdown",
            inline_code_theme="solarized-dark",
        )

        panel = Panel(markdown, **panel_style)

        if interaction.content:
            if interaction.role == Interaction.Role.USER:
                return Align.right(panel)
            elif interaction.role == Interaction.Role.ASSISTANT:
                return Align.left(panel)
            else:
                return Align.center(panel)

        return panel

    async def get_input(self, **kwargs) -> Interaction:
        """
        Gets user input from the command line.
        """
        exit_phrases = ["exit", "quit", "q", "quit()", "exit()"]
        clear_line = kwargs.pop("clear_line", False)
        default: str = kwargs.get("default", "")
        answer: str = default

        if kwargs.get("choices"):
            answer: str = await questionary.select(**kwargs).ask_async()
        else:
            answer: str = await questionary.text(**kwargs).ask_async()

        if answer is None or answer.lower() in exit_phrases:
            await self.exit_program()
            sys.exit(0)

        if clear_line:
            print("\033[A\033[K", end="")

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
            "expand": PANEL_EXPAND,
            "width": PANEL_WIDTH,
            "padding": (1, 2)
        }

        if (subtitle := output.subtitle):
            panel_style["subtitle"] = subtitle
        elif output.metadata.get("intention"):
            panel_style["subtitle"] = f"intention: {output.metadata['intention']}"

        if output.content:
            self.console.print(self.get_panel(output, **panel_style))

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

    def show_live_output(self, buffer: object, structured: object) -> None:
        """Show partial output."""

        assert isinstance(buffer, str)
        assert isinstance(structured, str)

        if not self.live:
            self.live = Live(
                console=self.console,
                refresh_per_second=15,
                auto_refresh=True,
                transient=True,
                vertical_overflow="visible",
            )
            self.live.start()

        panels = []

        if buffer and buffer.strip():
            scratchpad_panel = Panel(
                Markdown(buffer, inline_code_lexer="markdown", inline_code_theme="solarized-dark", style="bright white"),
                title=f"{Emoji('notebook')} Scratchpad",
                title_align="left",
                border_style="dim white",
                expand=PANEL_EXPAND,
                width=int(PANEL_WIDTH * 0.6),
                padding=(1, 2)
            )
            panels.append(scratchpad_panel)

        if structured and structured.strip():
            structured_panel = Panel(
                Markdown(structured, inline_code_lexer="json", inline_code_theme="solarized-dark"),
                title=f"{Emoji('gear')} Structured Output",
                title_align="left",
                border_style="cyan",
                expand=PANEL_EXPAND,
                width=int(PANEL_WIDTH * 0.8),
                padding=(1, 2)
            )
            panels.append(structured_panel)

        if panels:
            self.live.update(Group(*panels))

    def end_live_output(self) -> None:
        """End live output."""
        if self.live:
            self.console.print(self.live.renderable)
            self.live.stop()
            self.live = None
            self.console.clear_live()

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
        from imgcat import imgcat
        from PIL import Image

        try:
            img = Image.open(image_url)
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
            border_style = "white"

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
            padding=(1, 2)
        )
        self.console.print(Align.center(panel))

    async def clear(self) -> None:
        """
        Clears the terminal screen based on the operating system.
        """
        if sys.platform.startswith("win"):
            os.system("cls")  # For Windows
        else:
            os.system("clear")  # For Unix/Linux/macOS
