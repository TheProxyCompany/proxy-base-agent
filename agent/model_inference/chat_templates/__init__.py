import os

from .control_tokens import get_control_tokens


def load_chat_template(name: str = "chat_template.jinja") -> str:
    """Load the chat template from the specified file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, name)
    with open(template_path) as file:
        return file.read()


__all__ = ["get_control_tokens", "load_chat_template"]
