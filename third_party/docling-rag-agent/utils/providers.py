"""
Providers for LLM and embedding clients.
"""

import os
from functools import lru_cache

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


def _normalize_ollama_base_url(raw: str) -> str:
    """Ensure the Ollama base URL points at the OpenAI-compatible endpoint."""
    normalized = raw.rstrip('/')
    if not normalized.endswith('/v1'):
        normalized = f"{normalized}/v1"
    return normalized


@lru_cache(maxsize=1)
def get_llm_provider() -> OllamaProvider:
    """Return a cached Ollama provider instance."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    base_url = _normalize_ollama_base_url(base_url)
    return OllamaProvider(base_url=base_url)


def get_llm_model(model_name: str = "gemma3:1b") -> OpenAIChatModel:
    """Construct an OpenAI-compatible model backed by Ollama."""
    provider = get_llm_provider()
    return OpenAIChatModel(model_name, provider=provider)
