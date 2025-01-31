import os


def get_available_prompts() -> list[str]:
    """Get a list of available prompts."""
    prompt_file_names = []
    for file in os.scandir(os.path.dirname(__file__)):
        if file.is_file() and file.name.endswith(".txt"):
            prompt_file_names.append(file.name)
    return sorted(prompt_file_names)


def load_template(name: str = "chat_template.jinja") -> str:
    """
    Load the chat template from the specified file.
    Looks in the current directory (agent/prompts) for the file.

    Args:
        name: The name of the template file.
        Defaults to "chat_template.jinja".

    Returns:
        The content of the template file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, name)
    with open(template_path) as file:
        return file.read()


def load_prompt(filepath: str | None = None) -> str | None:
    """Load a prompt template from a .txt, .jinja, or .md file."""
    extensions = [".txt", ".jinja", ".md"]

    for ext in extensions:
        file_name = f"{filepath}{ext}"
        full_path = os.path.join(os.path.dirname(__file__), file_name)
        if os.path.exists(full_path):
            try:
                with open(full_path) as f:
                    template = f.read()
                    return template
            except FileNotFoundError:
                continue

    return None
