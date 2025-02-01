import json
import logging
import os

DEFAULT_MODEL_FOLDER = ".language_models"
DEFAULT_MODEL_NAME = "Llama-3.1-8B-Instruct"

logger = logging.getLogger(__name__)


def _get_model_paths() -> list[str]:
    """
    Get list of paths to search for models.

    Returns:
        list[str]: List of paths to search
    """
    paths = []

    # Add HuggingFace path
    hf_path = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface/hub/")
    if os.path.exists(hf_path):
        paths.append(hf_path)

    # Add default models path
    root_dir = os.path.dirname(__file__)
    default_path = f"{root_dir}/../../../{DEFAULT_MODEL_FOLDER}"
    if os.path.exists(default_path):
        paths.append(default_path)

    return paths


def _get_config_path(model_dir: os.DirEntry) -> tuple[str, str]:
    """
    Get the config path and model path for a model directory.

    Args:
        model_dir (os.DirEntry): Directory entry for the model

    Returns:
        tuple[str, str]: Tuple of (config_path, model_path)
    """
    model_path = model_dir.path
    config_path = f"{model_dir.path}/config.json"

    if not os.path.exists(config_path):
        snapshots_path = os.path.join(model_dir.path, "snapshots")
        if os.path.exists(snapshots_path):
            for snapshot_dir in os.scandir(snapshots_path):
                if snapshot_dir.is_dir():
                    config_path = os.path.join(snapshot_dir.path, "config.json")
                    model_path = snapshot_dir.path
                    break

    return config_path, model_path


def _scan_model_dir(model_dir: os.DirEntry) -> tuple[str, str, str] | None:
    """
    Scan a model directory and extract model information.

    Args:
        model_dir (os.DirEntry): Directory entry for the model

    Returns:
        tuple[str, str, str] | None: Tuple of (name, path, type) if valid model, None otherwise
    """
    if not model_dir.is_dir():
        return None

    model_name = model_dir.name
    config_path, model_path = _get_config_path(model_dir)

    try:
        if not os.path.exists(config_path):
            return None

        with open(config_path) as f:
            config: dict = json.load(f)
            if model_type := config.get("model_type"):
                return (model_name, model_path, model_type)
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")

    return None


def get_available_models() -> list[tuple[str, str, str]]:
    """
    Get a list of available models from all model paths.

    Returns:
        list[tuple[str, str, str]]: A list of available models (name, path, type)
    """
    model_file_names: list[tuple[str, str, str]] = []

    for path in _get_model_paths():
        for model_dir in os.scandir(path):
            if model_info := _scan_model_dir(model_dir):
                model_file_names.append(model_info)

    return sorted(model_file_names, key=lambda x: x[0])
