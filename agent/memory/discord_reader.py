"""Discord reader.

Note: this file is named discord_reader.py to avoid conflicts with the
discord.py module.

"""
import pytz
import os
import asyncio
from typing import List, Optional
from pydantic import Field

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

async def read_channels(
    discord_token: str,
    limit: Optional[int],
    oldest_first: bool,
) -> List[Document]:
    """Async read channel.

    This function connects to Discord, reads all messages from all accessible channels,
    and converts them into Document objects.
    """
    import discord

    messages: List[discord.Message] = []
    documents: List[Document] = []

    class CustomClient(discord.Client):
        async def on_ready(self):
            try:
                for guild in self.guilds:
                    for channel in guild.text_channels:
                        if self.get_channel(channel.id):
                            try:
                                await self.read_channel_messages(channel)
                                await self.read_thread_messages(channel)
                            except discord.Forbidden:
                                # the bot doesn't have permission to read this channel
                                continue
                            except Exception as e:
                                print(f"Error reading channel {channel.name}: {str(e)}")
                await self.close()
            except Exception as e:
                print(f"Encountered error: {str(e)}")

        async def read_channel_messages(self, channel: discord.TextChannel):
            async for msg in channel.history(limit=limit, oldest_first=oldest_first):
                messages.append(msg)

        async def read_thread_messages(self, channel: discord.TextChannel):
            for thread in channel.threads:
                async for msg in thread.history(limit=limit, oldest_first=oldest_first):
                    messages.append(msg)

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    client = CustomClient(intents=intents)

    await client.start(discord_token)

    print(f"Total messages read: {len(messages)}")

    eastern = pytz.timezone('America/New_York')
    # Convert messages to Documents

    documents = []
    for msg in messages:
        if msg.content is not None and msg.content != "":
            document = Document(text=msg.content)
            document.metadata = {
                "username": msg.author.name,
                "created_at": msg.created_at.astimezone(eastern).isoformat(),
                "channel": msg.channel.name if isinstance(msg.channel, discord.TextChannel) else "DM"
            }
            documents.append(document)

    return documents


class DiscordReader(BasePydanticReader):
    """Discord reader.

    Reads conversations from channels.

    Args:
        discord_token (Optional[str]): Discord token. If not provided, we
            assume the environment variable `DISCORD_TOKEN` is set.

    """

    discord_token: str = Field(
        default=os.environ.get("DISCORD_TOKEN"),
        description="Discord token.",
    )


    def __init__(self, discord_token: Optional[str] = None) -> None:
        """Initialize with parameters."""
        if discord_token is None:
            discord_token = os.environ["DISCORD_TOKEN"]
            if discord_token is None:
                raise ValueError(
                    "Must specify `discord_token` or set environment "
                    "variable `DISCORD_TOKEN`."
                )
        super().__init__(is_remote=True)
        self.discord_token = discord_token

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "DiscordReader"

    def load_data(
        self,
        limit: Optional[int] = None,
        oldest_first: bool = True,
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            channel_ids (List[int]): List of channel ids to read.
            limit (Optional[int]): Maximum number of messages to read.
            oldest_first (bool): Whether to read oldest messages first.
                Defaults to `True`.

        Returns:
            List[Document]: List of documents.

        """
        results: List[Document] = []
        documents = asyncio.run(
            read_channels(
                self.discord_token,
                limit,
                oldest_first)
        )
        results += documents
        return results
