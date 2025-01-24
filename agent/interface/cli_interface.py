import os
import sys

import questionary
from interface import Interface
from rich.align import Align, AlignMethod
from rich.emoji import Emoji
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# from agent import StepResult
from agent.message import Message, MessageState

MESSAGE_DISPLAY_THRESHOLD: int = 8


if os.getenv("LOG_LEVEL") == "DEBUG":
    MESSAGE_DISPLAY_THRESHOLD = 0
elif os.getenv("LOG_LEVEL") == "INFO":
    MESSAGE_DISPLAY_THRESHOLD = 5


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
        try:
            user_input: str | None = await questionary.text(
                message=prompt or "Enter your message [enter to send, Ctrl+C to exit]:",
                qmark=">",
                default="",
            ).ask_async()
            # Clear the line after input
            sys.stdout.write("\033[1A\033[2K\033[G")
            sys.stdout.flush()
        except KeyboardInterrupt:
            return None

        if not user_input:
            return None

        return Message(
            role="user",
            content=user_input.strip(),
            state=MessageState.USER_INPUT,
            name="User",
        )

    async def handle_output(self, input: object | list[object]) -> None:
        """Handles and displays different types of messages.

        Args:
            message: The `Message` object to be handled.
        """
        if isinstance(input, list):
            for message in input:
                await self.handle_output(message)
            return

        if not isinstance(input, Message):
            return

        if input.role == "user":
            await self._display_user_message(input)
        elif input.role == "assistant":
            if input.state == MessageState.TOOL_CALL:
                # scratch pad
                await self._display_assistant_message(
                    input, border_style="blue", emoji="spiral_notepad"
                )
            else:
                await self._display_assistant_message(input)
        elif input.role == "ipython":
            if input.state == MessageState.ASSISTANT_RESPONSE:
                await self._display_assistant_message(input)
            elif input.state == MessageState.METACOGNITION:
                await self._display_metacognition_message(input)
            elif input.state in [MessageState.TOOL_RESULT, MessageState.TOOL_ERROR]:
                await self._display_tool_result_message(input)
        elif input.role == "system":
            await self._display_system_message(input)
        else:
            await self._display_generic_message(input)

    async def _display_user_message(self, message: Message) -> None:
        """Displays a user message with a speech balloon emoji."""
        await self._display_message(
            message, message.name or "User", "green", "speech_balloon"
        )

    async def _display_assistant_message(
        self, message: Message, border_style: str = "green", emoji: str = "robot"
    ) -> None:
        """Displays an assistant message with a robot emoji."""
        await self._display_message(
            message=message,
            title=message.name or "Assistant",
            border_style=border_style,
            emoji=emoji,
        )

    async def _display_system_message(self, message: Message) -> None:
        """Displays a system message with a gear emoji."""
        await self._display_message(message, message.name or "System", "purple", "gear")

    async def _display_metacognition_message(self, message: Message) -> None:
        """Display thoughts with thought balloon and brain emoji."""
        monologue = f"*{message.inner_thoughts}*"
        subtitle_text = (
            f"{Emoji('brain')} Feeling: {message.feelings}"
            if message.feelings
            else None
        )
        self.console.print(
            Align.left(
                Panel(
                    Markdown(monologue, justify="left"),
                    title=f"{Emoji('thought_balloon')} {message.name}'s Thoughts",
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

    async def _display_tool_result_message(self, message: Message) -> None:
        """Displays the result (or error) of a function call."""
        if message.image_path:
            await self.show_image(message)
            return

        status = "Success" if message.state == MessageState.TOOL_RESULT else "Error"
        self.console.print(
            Align.left(
                Panel(
                    Markdown(str(message.content), justify="left"),
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

    async def _display_generic_message(self, message: Message) -> None:
        """Displays a generic message in a blue panel."""
        await self._display_message(
            message, f"Message ({message.role})", "blue", "warning"
        )

    async def _display_message(
        self,
        message: Message,
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
        if (
            message.content is None
            or message.content == ""
            or message.state.get_value() < MESSAGE_DISPLAY_THRESHOLD
        ):
            return
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
        self, message: Message | None = None, e: Exception | None = None
    ) -> None:
        """Display an error message with a warning emoji."""
        error_message = message or Message(role="system", content=f"{e}")
        await self._display_message(error_message, "Error", "red", "warning")

    async def show_image(self, image: object) -> None:
        """Displays an image from a URL, with optional caption and thoughts.

        Args:
            message: The `Message` object, which should include the
                      `image_path`,  `content` (caption), and `inner_thoughts`
                      if available.
        """
        from urllib.request import urlopen

        from imgcat import imgcat
        from PIL import Image

        if not isinstance(image, Message):
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
            Message(role="ipython", content=content.strip()),
            "Mind's Eye",
            "cyan",
            "framed_picture",
        )

    async def exit_program(self) -> None:
        """Exits the program with a goodbye message and a waving hand emoji."""
        self.console.print(f"{Emoji('wave')} [bold]Goodbye![/bold]")

    def _print_to_console(
        self,
        text: str,
        title: str,
        border_style: str,
        emoji: str,
        justify: AlignMethod = "left",
    ) -> None:
        """Print a message to the console."""
        self.console.print(
            Align.left(
                Panel(
                    Markdown(markup=text, justify=justify),
                    title=Text(
                        text=f"{Emoji(emoji)} {title}", style="italic", justify=justify
                    ),
                    title_align=justify,
                    subtitle_align=justify,
                    border_style=border_style,
                    expand=self.PANEL_EXPAND,
                    width=self.PANEL_WIDTH,
                )
            )
        )

    async def clear(self) -> None:
        """
        Clears the terminal screen based on the operating system.
        """
        if sys.platform.startswith("win"):
            os.system("cls")  # For Windows
        else:
            os.system("clear")  # For Unix/Linux/macOS
