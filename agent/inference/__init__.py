import json
import logging
import os

from agent.inference.frontend import FrontEnd, FrontEndType

logger = logging.getLogger(__name__)

DEFAULT_MODEL_FOLDER = ".language_models"
DEFAULT_MODEL_NAME = "Llama-3.1-8B-Instruct"

def get_available_models() -> list[tuple[str, str, str]]:
    """
    Get a list of available models.

    Returns:
        list[tuple[str, str, str]]: A list of available models (name, path, type)
    """
    path = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface/hub/")

    if not os.path.exists(path):
        root_dir = os.path.dirname(__file__)
        path = f"{root_dir}/../../{DEFAULT_MODEL_FOLDER}"
        breakpoint()
        if not os.path.exists(path):
            return []

    model_file_names: list[tuple[str, str, str]] = []
    for model_dir in os.scandir(path):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        model_path = f"{path}/{model_dir.name}"
        model_config_path = f"{model_dir.path}/config.json"

        try:
            if not os.path.exists(model_config_path):
                snapshots_path = os.path.join(model_dir.path, "snapshots")
                if os.path.exists(snapshots_path):
                    # Check for valid snapshot configuration for a HuggingFace model
                    for model_snapshot_dir in os.scandir(snapshots_path):
                        if model_snapshot_dir.is_dir():
                            model_config_path = os.path.join(model_snapshot_dir.path, "config.json")
                            model_path = model_snapshot_dir.path
                            break

            if not os.path.exists(model_config_path):
                continue
            with open(model_config_path) as f:
                config: dict = json.load(f)
                if model_type := config.get("model_type"):
                    model_file_names.append((model_name, model_path, model_type))
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            continue

    return sorted(model_file_names, key=lambda x: x[0])


__all__ = ["FrontEnd", "FrontEndType", "get_available_models"]
