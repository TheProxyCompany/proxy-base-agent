import logging
import os
import sys

import questionary
from rich.align import Align
from rich.console import RenderableType
from rich.emoji import Emoji
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from agent.event import Event, State
from agent.interface import Interface

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
            role="user",
            content=user_input.strip(),
            state=State.USER_INPUT,
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

        if input.role == "user":
            await self._display_user_message(input)
        elif input.role == "assistant":
            if input.state == State.TOOL_CALL:
                # scratch pad
                await self._display_assistant_message(
                    input, border_style="blue", emoji="spiral_notepad"
                )
            else:
                await self._display_assistant_message(input)
        elif input.role == "ipython":
            if input.state == State.ASSISTANT_RESPONSE:
                await self._display_assistant_message(input)
            elif input.state == State.METACOGNITION:
                await self._display_metacognition_message(input)
            elif input.state in [State.TOOL_RESULT, State.TOOL_ERROR]:
                await self._display_tool_result_message(input)
        elif input.role == "system":
            await self._display_system_message(input)
        else:
            await self._display_generic_message(input)

    async def show_live_output(self, output: object) -> None:
        """Show partial output."""

        self.live_content += str(output)
        self.render_panel(refresh_per_second=30)

    async def _display_user_message(self, message: Event) -> None:
        """Displays a user message with a speech balloon emoji."""
        await self._display_message(
            message, message.name or "User", "green", "speech_balloon"
        )

    async def _display_assistant_message(
        self, message: Event, border_style: str = "green", emoji: str = "robot"
    ) -> None:
        """Displays an assistant message with a robot emoji."""
        await self._display_message(
            message=message,
            title=message.name or "Assistant",
            border_style=border_style,
            emoji=emoji,
        )

    async def _display_system_message(self, message: Event) -> None:
        """Displays a system message with a gear emoji."""
        await self._display_message(message, message.name or "System", "purple", "gear")

    async def _display_metacognition_message(self, message: Event) -> None:
        """Display thoughts with thought balloon and brain emoji."""
        subtitle_text = (
            f"{Emoji('brain')} Feeling: {message.feelings}"
            if message.feelings
            else None
        )

        self.console.print(
            Align.left(
                Panel(
                    Markdown(str(message.content), justify="left"),
                    title=f"{Emoji('thought_balloon')} Thoughts",
                    title_align="left",
                    subtitle=subtitle_text,
                    subtitle_align="left",
                    border_style="magenta",
                    expand=self.PANEL_EXPAND,
                    width=self.PANEL_WIDTH,
                )
            )
        )
        self.console.print()

    async def _display_tool_result_message(self, message: Event) -> None:
        """Displays the result (or error) of a function call."""
        if message.image_path:
            breakpoint()
            await self.render_image(message)
            return

        status = "Success" if message.state == State.TOOL_RESULT else "Error"
        content = message.content
        if isinstance(content, list):
            content = "\n".join(content)
        self.console.print(
            Align.left(
                Panel(
                    Markdown(content, justify="left"),
                    title=f"{Emoji('zap')} {message.name}",
                    title_align="left",
                    subtitle_align="left",
                    border_style="green" if status == "Success" else "red",
                    expand=self.PANEL_EXPAND,
                    width=self.PANEL_WIDTH,
                )
            )
        )
        self.console.print()

    async def _display_generic_message(self, message: Event) -> None:
        """Displays a generic message in a blue panel."""
        await self._display_message(
            message, f"Message ({message.role})", "blue", "warning"
        )

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

    async def render_image(self, image: object) -> None:
        """Displays an image from a URL, with optional caption and thoughts.

        Args:
            message: The `Message` object, which should include the
                      `image_path`,  `content` (caption), and `inner_thoughts`
                      if available.
        """
        from urllib.request import urlopen

        from imgcat import imgcat
        from PIL import Image

        if not isinstance(image, Event):
            return

        if image.image_path:
            try:
                img = Image.open(urlopen(image.image_path))
                imgcat(img)
            except Exception as error:
                await self.show_error_message(e=error)

        content = f"\n[Link to full image]({image.image_path})\n\n"
        if image.content:
            content += f"### {Emoji('label')} Caption\n*{image.feelings}*\n\n"
        if image.inner_thoughts:
            content += f"### {Emoji('thought_balloon')} Inner Thoughts\n*{image.inner_thoughts}*"

        await self._display_message(
            Event(role="ipython", content=content.strip()),
            "Mind's Eye",
            "cyan",
            "framed_picture",
        )

    async def exit_program(self, e: Exception | None = None) -> None:
        """Exits the program with a goodbye message and a waving hand emoji."""
        await self.show_error_message(e=e)
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

    def render_panel(
        self,
        content: str | None = None,
        title: str = "",
        border_style: str = "green",
        refresh_per_second: int = 30,
        **kwargs,
    ) -> None:
        """
        Display output with customizable panel settings and support for animated/partial updates.

        Args:
            content: The content to display.
            title: The title of the panel
            emoji: The emoji to display next to the title
            border_style: The style of the panel's border
            refresh_per_second: Number of times to refresh the live display per second
            **kwargs: Additional keyword arguments to override default settings
        """
        if not self.live:

            def get_renderable() -> RenderableType:
                markdown = Markdown(
                    content or self.live_content,
                    justify="left",
                    code_theme="monokai",
                    inline_code_lexer="markdown",
                    inline_code_theme="solarized-dark",
                )
                panel = Panel(
                    markdown,
                    title=title,
                    title_align="left",
                    border_style=border_style,
                    expand=True,
                )
                return panel

            self.live = Live(
                console=self.console,
                refresh_per_second=refresh_per_second,
                get_renderable=get_renderable,
            )
            self.live.start()
