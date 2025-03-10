import logging
import threading
from collections.abc import Iterator

import sounddevice as sd
from kokoro_onnx import Kokoro

logger = logging.getLogger(__name__)
class VoiceBox:
    def __init__(self):
        from agent.system.voice import MODEL_PATH, VOICES_PATH

        if not MODEL_PATH or not VOICES_PATH:
            download_path = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
            raise FileNotFoundError(
                f"Model or voices file not found. Download from {download_path} into the `system/voice/models` directory."
            )
        self.kokoro = Kokoro(MODEL_PATH, VOICES_PATH)

    def __call__(self, message: str, voice: str = "af_heart", lang: str = "en-us"):
        """
        Speak a message in a new thread.
        """
        self.speak(message, voice, lang)

    def speak(self, message: str, voice: str = "af_heart", lang: str = "en-us"):
        """Speak a message in a new thread."""
        thread = threading.Thread(
            target=self._speak_in_thread,
            args=(message, voice, lang)
        )
        thread.start()

    def _speak_in_thread(self, message: str, voice: str, lang: str):
        for chunk in self._clean_transcript(message):
            if not chunk:
                continue

            samples, sample_rate = self.kokoro.create(
                chunk,
                voice=voice,
                speed=1.3,
                lang=lang,
            )
            try:
                sd.play(samples, sample_rate)
                sd.wait()
            except sd.PortAudioError as e:
                logger.error(f"Fallback audio playback failed: {e}")
                pass

    def _clean_transcript(self, raw: str) -> Iterator[str]:
        """Clean message text for speech synthesis by removing unwanted characters."""
        text = raw.strip()
        # Remove unicode characters by encoding to ascii and back, ignoring errors
        text = text.encode("ascii", "ignore").decode("ascii", "ignore")
        yield text.strip()
