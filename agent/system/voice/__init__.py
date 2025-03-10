import os

VOICES_ROOT_FOLDER = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(VOICES_ROOT_FOLDER, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(VOICES_ROOT_FOLDER, "voices-v1.0.bin")

from agent.system.voice.voicebox import VoiceBox  # noqa: E402

__all__ = ["VoiceBox"]
