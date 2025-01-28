import json
import logging
import os

from agent.inference.frontend import FrontEnd, FrontEndType

DEFAULT_MODEL_FOLDER = ".language_models"
DEFAULT_MODEL_NAME = "Llama-3.1-SuperNova-Lite"


logger = logging.getLogger(__name__)

def get_available_models() -> list[tuple[str, str]]:
    """Get a list of available models."""
    model_file_names: list[tuple[str, str]] = []
    for model_dir in os.scandir(DEFAULT_MODEL_FOLDER):
        if not model_dir.is_dir():
            continue

        config_path = os.path.join(model_dir.path, "config.json")
        if not os.path.exists(config_path):
            continue

        try:
            with open(config_path) as f:
                config: dict = json.load(f)
                if model_type := config.get("model_type"):
                    model_file_names.append((model_dir.name, model_type))
        except Exception as e:
            logger.error(f"Error loading model {model_dir.name}: {e}")
            continue

    return sorted(model_file_names, key=lambda x: x[0])


__all__ = ["FrontEnd", "FrontEndType", "get_available_models"]
