import os
from collections.abc import Iterator


def get_available_prompts() -> Iterator[str]:
    """Get a list of available prompts."""
    for file in os.scandir(os.path.dirname(__file__)):
        if file.is_file() and file.name.endswith(".txt"):
            yield file.name


def load_prompt_template(filepath: str | None = None) -> str:
    """Load a prompt template from a .txt, .jinja, or .md file."""
    extensions = [".txt", ".jinja", ".md"]

    for ext in extensions:
        full_path = os.path.join(os.path.dirname(__file__), f"{filepath}{ext}")
        if os.path.exists(full_path):
            try:
                with open(full_path) as f:
                    template = f.read()
                    return template
            except FileNotFoundError:
                continue

    raise ValueError(f"valid prompt file not found in {filepath}")
