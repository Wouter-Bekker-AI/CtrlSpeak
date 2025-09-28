# SocialRobot-main/audio/remote_stt.py
from __future__ import annotations
import io
import wave
import requests

class RemoteSTT:
    """Posts WAV audio to CtrlSpeak’s /transcribe endpoint and returns text."""
    def __init__(self, url: str) -> None:
        if not url.endswith("/transcribe"):
            if url.endswith("/"):
                url = url + "transcribe"
            else:
                url = url + "/transcribe"
        self.url = url

    def run_stt(self, raw_bytes: bytes, sample_rate: int = 16000) -> str:
        if not raw_bytes:
            return ""
        # Convert raw 16-bit PCM mono bytes to a small WAV for the server
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(raw_bytes)
        data = buf.getvalue()
        r = requests.post(self.url, data=data, headers={"Content-Type": "audio/wav"}, timeout=60)
        r.raise_for_status()
        payload = r.json()
        return (payload.get("text") or "").strip()