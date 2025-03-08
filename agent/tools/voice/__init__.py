import os

from agent.tools.voice.voicebox import VoiceBox

try:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/kokoro-v1.0.onnx")
    VOICES_PATH = os.path.join(os.path.dirname(__file__), "models/voices-v1.0.bin")
except FileNotFoundError:
    MODEL_PATH = ""
    VOICES_PATH = ""

__all__ = ["VoiceBox"]
