"""Simple Ollama client helper."""

from __future__ import annotations

import json
from typing import Any, List, Optional

import requests


class OllamaClient:
    def __init__(
        self,
        url: str = "http://localhost:11434/api/chat",
        model: str = "gemma3:270m",
        stream: bool = True,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.url = url
        self.model = model
        self.stream = stream
        self.system_prompt = system_prompt

    def _normalize_history_entry(self, entry: dict) -> dict:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        return {"role": role, "content": content}

    def _build_messages(
        self,
        user_payload: Any,
        history: Optional[List[dict]] = None,
    ) -> List[dict]:
        messages: List[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if history:
            messages.extend(self._normalize_history_entry(item) for item in history)
        messages.append({"role": "user", "content": user_payload})
        return messages

    def _fallback_response(self, user_text: str, stream: bool, error: Exception | None = None):
        clean = user_text.strip()
        fallback = f"I heard you say: {clean}" if clean else "I'm here and listening."
        if error is not None:
            print("-> Ollama unavailable, using fallback response.")
        if stream:
            def generator():
                yield fallback
            return generator()
        return fallback

    def unload(self):
        """Unload the model from Ollama to free up GPU memory."""
        try:
            requests.post(self.url.replace("/chat", "/generate"), json={"model": self.model, "keep_alive": 0}, timeout=10)
            print(f"-> Requested Ollama to unload model: {self.model}")
        except Exception as exc:
            print(f"-> Failed to request Ollama to unload model: {exc}")

    def query(
        self,
        user_text: str,
        stream: Optional[bool] = None,
        history: Optional[List[dict]] = None,
        content: Any | None = None,
    ):
        use_stream = self.stream if stream is None else stream
        user_payload = content if content is not None else user_text
        payload = {
            "model": self.model,
            "messages": self._build_messages(user_payload, history=history),
            "stream": use_stream,
        }
        print("-> Sending user text to Ollama:\n", user_text)

        try:
            response = requests.post(self.url, json=payload, stream=use_stream, timeout=120)
            response.raise_for_status()
        except Exception as exc:
            return self._fallback_response(user_text, use_stream, exc)

        if not use_stream:
            data = response.json()
            message = data.get("message", {})
            response_text = message.get("content", "")
            return response_text.strip()

        def stream_generator():
            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    if chunk.get("done"):
                        break
                    message = chunk.get("message", {})
                    content = message.get("content")
                    if content:
                        yield content
            finally:
                response.close()

        return stream_generator()


__all__ = ["OllamaClient"]
