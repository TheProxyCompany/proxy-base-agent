
import random
import threading
from collections.abc import Iterator

import sounddevice as sd
from kokoro_onnx import Kokoro
from phonemizer.backend import BACKENDS


class VoiceBox:

    def __init__(self):
        from agent.voice import MODEL_PATH, VOICES_PATH
        if not MODEL_PATH or not VOICES_PATH:
            download_path = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
            raise FileNotFoundError(f"Model or voices file not found. Download from {download_path} into the voice/models directory.")
        self.kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
        BACKENDS["espeak"](language="en-us", words_mismatch="ignore")  # type: ignore reportCallIssue

    def __call__(self, message: str, voice: str = "af_bella", lang: str = "en-us"):
        """
        Speak a message in a new thread.
        """
        self.speak(message, voice, lang)

    def speak(self, message: str, voice: str = "af_bella", lang: str = "en-us"):
        """Speak a message in a new thread."""
        thread = threading.Thread(target=self._speak_in_thread, args=(message, voice, lang))
        thread.start()

    def _speak_in_thread(self, message: str, voice: str, lang: str):
        for chunk in self._clean_transcript(message):
            if not chunk:
                continue

            samples, sample_rate = self.kokoro.create(
                chunk,
                voice=voice,
                speed=random.uniform(0.9, 1.3),
                lang=lang,
            )
            sd.play(samples, sample_rate)
            sd.wait()

    def _clean_transcript(self, raw: str) -> Iterator[str]:
        """Clean message text for speech synthesis by removing unwanted characters."""
        for chunk in raw.replace("\n", ". ").replace("! ", ". ").replace("? ", ". ").split(". "):
            text = chunk.strip()
            text = " ".join(text.split())  # Collapse multiple spaces
            text = text.lstrip("-")  # Remove leading dashes
            # Remove unicode characters by encoding to ascii and back, ignoring errors
            text = text.encode("ascii", "ignore").decode("ascii", "ignore")
            yield text.strip()
