import logging
import os
import sys
import traceback
from typing import Any

import questionary
from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.emoji import Emoji
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from agent.interface import Interface
from agent.state import AgentState
from agent.system.interaction import Interaction

logger = logging.getLogger(__name__)

# Enhanced styling constants
PANEL_WIDTH = 90
PANEL_PADDING = (1, 1)


class CLIInterface(Interface):
    """An elegant command-line interface for interacting with the Brain agent.

    This class implements the AgentInterface and provides methods for
    displaying different message types with consistent, beautiful formatting
    using Rich library elements (Panels, Markdown, Emojis).
    """

    def __init__(self) -> None:
        self.console = Console()
        self.live: Live | None = None
        self.current_state: AgentState | None = None

    @staticmethod
    def get_panel(interaction: Interaction, **panel_style: Any) -> RenderableType:
        match interaction.content:
            case list():
                content = "\n\n".join(interaction.content)
            case _:
                content = str(interaction.content)

        # Enhanced markdown styling
        markdown = Markdown(
            content,
            justify="left",
            code_theme="monokai",
            inline_code_lexer="markdown",
            inline_code_theme="monokai",
            style="bright_white",
        )

        # Default box style based on role
        box_style = (
            box.SQUARE
            if interaction.role == Interaction.Role.USER
            else box.HEAVY
            if interaction.role == Interaction.Role.ASSISTANT
            else box.DOUBLE
        )

        panel = Panel(
            markdown,
            box=box_style,
            style=Style(color=panel_style.get("style", "bright_white")),
            **panel_style,
        )

        if not interaction.content:
            return panel

        # Align panel based on role
        if interaction.role == Interaction.Role.USER:
            panel.title_align = panel.subtitle_align = "right"
            return Align.right(panel)
        elif interaction.role == Interaction.Role.SYSTEM:
            panel.title_align = panel.subtitle_align = "center"
            return Align.center(panel)
        else:
            panel.title_align = panel.subtitle_align = "left"
            return Align.left(panel)

    async def get_input(self, **kwargs: Any) -> Interaction:
        """Gets user input with enhanced styling and validation."""
        exit_phrases = frozenset({"exit", "quit", "q", "quit()", "exit()"})
        clear_line = kwargs.pop("clear_line", False)

        # Enhanced questionary styling
        style = questionary.Style(
            [
                ("qmark", "fg:ansimagenta bold"),
                ("question", "fg:ansicyan bold"),
                ("answer", "fg:ansigreen bold"),
                ("pointer", "fg:ansimagenta bold"),
                ("highlighted", "fg:ansigreen bold"),
                ("selected", "fg:ansigreen bold"),
                ("separator", "fg:ansigray"),
                ("instruction", "fg:ansigray"),
            ]
        )

        kwargs["style"] = style

        answer = await (
            questionary.select(**kwargs).ask_async()
            if kwargs.get("choices")
            else questionary.text(**kwargs).ask_async()
        )

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
        """Displays messages with enhanced visual styling."""
        if not isinstance(output, Interaction):
            return

        if output.image_url:
            await self.render_image(output.image_url)

        style = {
            "color": output.styling.get("color", "info"),
            "title": output.styling.get("title", "") or output.title,
        }
        if "emoji" in output.styling:
            style["emoji"] = output.styling["emoji"]

        emoji = f"{Emoji(style['emoji'])} " if "emoji" in style else ""

        panel_style = {
            "border_style": style["color"],
            "title": f"{emoji}{style['title']}",
            "width": PANEL_WIDTH,
            "padding": PANEL_PADDING,
        }

        match output:
            case Interaction(subtitle=subtitle) if subtitle:
                panel_style["subtitle"] = subtitle
            case Interaction(metadata={"intention": intention}):
                panel_style["subtitle"] = f"intention: {intention}"

        if output.content:
            self.console.print(self.get_panel(output, **panel_style))

        if isinstance(output.tool_result, Interaction):
            await self.show_output(output.tool_result)

    def show_live_output(self, state: AgentState | None, output: object) -> None:
        """Show partial output with enhanced visual styling."""

        if state != self.current_state:
            if self.current_state is not None:
                self.end_live_output()
            self.current_state = state

        if not self.current_state:
            return

        if string_output := str(output).strip():
            if not self.live:
                self.live = Live(
                    console=self.console,
                    refresh_per_second=15,
                    auto_refresh=True,
                    transient=True,
                    vertical_overflow="visible",
                )
                self.live.start()

            structured_panel = Panel(
                Markdown(
                    self.current_state.readable_format(string_output),
                    inline_code_theme="monokai",
                    style="bright_white",
                ),
                title=f"{Emoji(self.current_state.emoji)} {self.current_state.readable_name.title()}",
                title_align="left",
                subtitle=Text(
                    "Powered by: The Proxy Structuring Engine",
                    style="bright_white italic",
                ),
                subtitle_align="left",
                border_style=self.current_state.color,
                width=PANEL_WIDTH,
                padding=(0, 0),
            )
            self.live.update(Align.left(structured_panel))

    def end_live_output(self) -> None:
        """End live output with a smooth transition."""
        if self.live:
            self.console.print(self.live.renderable)
            self.live.stop()
            self.live.update(Group())
            self.console.clear_live()
            self.live = None
        elif self.current_state:
            self.current_state = None
            self.console.print()

    async def show_error_message(
        self,
        message: Interaction | None = None,
        e: Exception | None = None,
    ) -> None:
        """Display an error message with enhanced styling and helpful context."""
        if not message and not e:
            return

        error_message = message
        if e:
            error_content = [
                f"*{e.__class__.__name__}*: {e}",
                "",
                "```python",
                traceback.format_exc(),
                "```",
            ]
            error_message = Interaction(
                role=Interaction.Role.SYSTEM,
                content="\n".join(error_content),
                styling={"color": "error", "emoji": "warning"},
            )

        if error_message and error_message.content:
            panel_style = {
                "border_style": "red",
                "title": f"{Emoji('warning')} Error Details",
                "subtitle": "Please check the information below",
                "padding": (0, 0),
                "box": box.HEAVY,
            }

            self.console.print()
            self.console.print(self.get_panel(error_message, **panel_style))
            self.console.print()

    async def render_image(self, image_url: str) -> None:
        """Displays an image with enhanced error handling."""
        try:
            from imgcat import imgcat
            from PIL import Image

            img = Image.open(image_url)
            imgcat(img)
        except ImportError:
            await self.show_error_message(
                Interaction(
                    role=Interaction.Role.SYSTEM,
                    content="Required packages 'imgcat' or 'Pillow' not found. Please install them to display images.",
                    styling={"color": "error", "emoji": "warning"},
                )
            )
        except Exception as e:
            await self.show_error_message(e=e)

    async def exit_program(self, error: Exception | None = None) -> None:
        """Exits the program with a stylish goodbye message."""
        if error:
            title = f"{Emoji('exclamation')} Error Occurred"
            content = f"```pytb\n{traceback.format_exc()}\n```"
            border_style = "red"
        else:
            title = f"{Emoji('wave')} Goodbye"
            content = "*See you again!*"
            border_style = "blue"
        markdown = Markdown(
            content,
            justify="center",
            inline_code_theme="one-dark",
        )

        panel = Panel(
            markdown,
            title=title,
            title_align="center",
            border_style=border_style,
            box=box.DOUBLE,
            expand=False,
            width=PANEL_WIDTH,
            padding=(0, 0),
        )

        self.console.print()
        self.console.print(Align.center(panel))
        self.console.print()

    async def clear(self) -> None:
        """
        Clears the terminal screen based on the operating system.
        """
        if sys.platform.startswith("win"):
            os.system("cls")  # For Windows
        else:
            os.system("clear")  # For Unix/Linux/macOS
