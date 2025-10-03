"""Simple Ollama client helper."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import requests


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama service cannot be reached."""

    def __init__(self, message: str, original_exception: Exception | None = None) -> None:
        super().__init__(message)
        self.original_exception = original_exception


class OllamaClient:
    def __init__(
        self,
        url: str = "http://localhost:11434/api/chat",
        model: str = "gemma3:270m",
        stream: bool = True,
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        hardware_mode: Optional[str] = None,
    ) -> None:
        self.url = url
        self.model = model
        self.stream = stream
        self.system_prompt = system_prompt
        self._base_options = dict(options) if options else {}
        self._hardware_mode = (hardware_mode or "").strip().lower() or None

    @staticmethod
    def _split_content_blocks(blocks: List[dict]) -> Tuple[str, List[str]]:
        """Convert mixed text/image blocks into Ollama's chat format."""

        text_parts: List[str] = []
        images: List[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "").lower()
            if block_type == "text":
                text_value = block.get("text")
                if text_value:
                    text_parts.append(str(text_value))
            elif block_type == "image":
                image_data = block.get("image")
                if not image_data:
                    continue
                image_str = str(image_data)
                if image_str.startswith("data:"):
                    try:
                        image_str = image_str.split(",", 1)[1]
                    except IndexError:
                        continue
                images.append(image_str)
        text_content = "\n\n".join(part.strip() for part in text_parts if part).strip()
        return text_content, images

    def _compose_options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {k: v for k, v in self._base_options.items() if v is not None}
        mode = (self._hardware_mode or "").lower()

        if mode == "cpu_only":
            options["num_gpu"] = 0
        elif mode == "gpu_only":
            options["num_gpu"] = options.get("num_gpu", 999)
            options.setdefault("main_gpu", 0)
            options.setdefault("gpu_split", [0.6, 0.4])
            options.setdefault("low_vram", False)
        else:
            if "num_gpu" not in options:
                options["num_gpu"] = None

        return {k: v for k, v in options.items() if v is not None}

    def _ps(self) -> dict:
        endpoint = self.url
        if endpoint.endswith("/chat"):
            endpoint = f"{endpoint[:-5]}/ps"
        else:
            endpoint = endpoint.replace("/chat", "/ps")
        try:
            response = requests.get(endpoint, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _ensure_gpu_only(self) -> None:
        info = self._ps()
        models = info.get("models") if isinstance(info, dict) else []
        target = self.model.split(":", 1)[0]

        for model_info in models:
            name = str(model_info.get("model") or model_info.get("name") or "")
            if name and target not in name:
                continue

            if model_info.get("gpu") is False:
                raise OllamaUnavailableError(
                    "Strict GPU mode requires the model to reside entirely on the GPU (CPU offload detected)."
                )

            vram = model_info.get("size_vram") or model_info.get("vram")
            size = model_info.get("size") or model_info.get("size_model")
            if size and not vram:
                raise OllamaUnavailableError(
                    "Strict GPU mode requires the model to reside entirely on the GPU (missing VRAM allocation)."
                )

        # If the model list is empty or the model isn't present, assume Ollama hasn't loaded it yet.
        # The subsequent query will surface availability errors if the preload failed.

    def _normalize_history_entry(self, entry: dict) -> dict:
        role = entry.get("role", "user")
        content = entry.get("content", "")

        if isinstance(content, list):
            text_content, images = self._split_content_blocks(content)
            message: Dict[str, Any] = {"role": role, "content": text_content}
            if images:
                message["images"] = images
            return message

        return {"role": role, "content": content}

    def _build_user_message(self, user_payload: Any) -> dict:
        if isinstance(user_payload, dict):
            return self._normalize_history_entry(user_payload)
        if isinstance(user_payload, list):
            text_content, images = self._split_content_blocks(user_payload)
            message: Dict[str, Any] = {"role": "user", "content": text_content}
            if images:
                message["images"] = images
            return message
        return {"role": "user", "content": user_payload}

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
        messages.append(self._build_user_message(user_payload))
        return messages

    def _fallback_response(self, user_text: str, stream: bool, error: Exception | None = None):
        error_details = str(error) if error else "Unknown error"
        message = f"I couldn't reach Ollama: {error_details}"
        print(f"-> Ollama unavailable: {error_details}")
        raise OllamaUnavailableError(message, error)

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
        options = self._compose_options()
        if options:
            payload["options"] = options

        if self._hardware_mode == "gpu_only":
            preload_payload = {"model": self.model, "messages": [], "stream": False}
            if options:
                preload_payload["options"] = options
            try:
                preload_response = requests.post(self.url, json=preload_payload, timeout=60)
                preload_response.raise_for_status()
            except Exception as exc:
                return self._fallback_response(user_text, use_stream, exc)
            try:
                self._ensure_gpu_only()
            except OllamaUnavailableError:
                self.unload()
                raise
            except Exception as exc:
                self.unload()
                return self._fallback_response(user_text, use_stream, exc)

        print("-> Sending user text to Ollama:\n", user_text)

        try:
            response = requests.post(self.url, json=payload, stream=use_stream, timeout=120)
            response.raise_for_status()
        except requests.HTTPError as exc:
            error_detail = ""
            if exc.response is not None:
                try:
                    error_detail = exc.response.text.strip()
                except Exception:
                    error_detail = ""
            if error_detail:
                error_message = f"{exc} - {error_detail}"
                exc = requests.HTTPError(error_message, response=exc.response, request=exc.request)
            return self._fallback_response(user_text, use_stream, exc)
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


    def chat(
        self,
        messages: List[dict],
        *,
        stream: Optional[bool] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        use_stream = self.stream if stream is None else stream
        if use_stream:
            raise ValueError("Streaming is not supported for chat() tool routing calls.")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": use_stream,
        }
        options = self._compose_options()
        if options:
            payload["options"] = options
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        if self._hardware_mode == "gpu_only":
            preload_payload = {"model": self.model, "messages": [], "stream": False}
            if options:
                preload_payload["options"] = options
            try:
                preload_response = requests.post(self.url, json=preload_payload, timeout=60)
                preload_response.raise_for_status()
            except Exception as exc:
                self._fallback_response("", False, exc)
            try:
                self._ensure_gpu_only()
            except OllamaUnavailableError:
                self.unload()
                raise
            except Exception as exc:
                self.unload()
                raise OllamaUnavailableError(str(exc)) from exc

        probe_text = ""
        for entry in reversed(messages):
            if isinstance(entry, dict) and entry.get("role") == "user":
                probe_text = str(entry.get("content") or "")
                break

        try:
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status()
        except requests.HTTPError as exc:
            error_detail = ""
            if exc.response is not None:
                try:
                    error_detail = exc.response.text.strip()
                except Exception:
                    error_detail = ""
            if error_detail:
                error_message = f"{exc} - {error_detail}"
                exc = requests.HTTPError(error_message, response=exc.response, request=exc.request)
            self._fallback_response(probe_text, use_stream, exc)
        except Exception as exc:
            self._fallback_response(probe_text, use_stream, exc)

        data = response.json()
        message = data.get("message") if isinstance(data, dict) else None
        if isinstance(message, dict):
            return {"message": message, "raw": data}
        return {"message": {}, "raw": data}


__all__ = ["OllamaClient", "OllamaUnavailableError"]
