from typing import List, Dict, Tuple
from .discord_reader import DiscordReader
from llama_index.core import Document
from collections import defaultdict
import os
from config import get_config


def get_team_discord_messages() -> Tuple[List[Document], int]:
    """
    Retrieve Discord messages, group them by channel, and create documents.

    Returns:
        Tuple[List[Document], int]: A list of Document objects containing Discord messages and the total count of processed messages.
    """
    discord_token = os.getenv("DISCORD_TOKEN")
    if not discord_token:
        raise ValueError("DISCORD_TOKEN environment variable is not set")
    discord_reader = DiscordReader(discord_token)
    raw_documents: List[Document] = discord_reader.load_data()

    documents: List[Document] = []
    channel_messages: Dict[str, List[Document]] = defaultdict(list)
    message_count = 0

    # print(f"raw_documents: {raw_documents[0:10]}")

    for doc in raw_documents:
        channel_name = doc.metadata.get("channel")
        if channel_name:
            channel_messages[channel_name].append(doc)
            message_count += 1

    # Process messages for each channel
    for channel_name, msgs in channel_messages.items():
        documents.append(_create_document(msgs, channel_name))

    return documents, message_count

def _create_document(messages: List[Document], channel_name: str) -> Document:
    """
    Helper function to create a document from a list of messages.

    Args:
        messages (List[Document]): A list of Document objects containing Discord messages.
        channel_name (str): The name of the channel.

    Returns:
        Document: A single Document object containing the combined messages.
    """
    combined_text = f"Channel: {channel_name}\n\n"
    team_map: Dict[str, str] = get_config().discord.team_map

    for msg in messages:
        username = team_map.get(msg.metadata.get("username") or "unknown", msg.metadata.get("username", "unknown"))
        combined_text += (
            f"Author: {username}\n"
            f"Date: {msg.metadata.get('created_at')}\n"
            f"Message: {msg.text}\n\n"
        )

    return Document(text=combined_text.strip())
