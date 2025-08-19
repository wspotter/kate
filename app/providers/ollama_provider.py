"""Ollama provider integration for local LLM models.

Minimal implementation that queries the local Ollama HTTP API.
"""
from __future__ import annotations

import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from loguru import logger

from .base import (
    BaseLLMProvider,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ProviderStatus,
)

OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"

class OllamaProvider(BaseLLMProvider):
    """Local Ollama provider.

    Exposes a subset of the chat completion style interface using Ollama's
    /api/chat (for non-stream) and /api/chat with stream=true endpoints.
    """
    def __init__(self, name: str = "ollama", config: Dict[str, Any] | None = None):
        super().__init__(name, config or {})
        self.base_url = self.config.get("base_url", OLLAMA_DEFAULT_URL)
        self._client: Optional[httpx.AsyncClient] = None
        self._models_cache: Optional[List[ModelInfo]] = None

    async def connect(self) -> bool:  # type: ignore[override]
        if self.status in (ProviderStatus.CONNECTED, ProviderStatus.CONNECTING):
            return True
        self.update_status(ProviderStatus.CONNECTING)
        try:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30)
            # Simple ping by listing models
            await self.get_models()
            self.update_status(ProviderStatus.CONNECTED)
            return True
        except Exception as e:
            logger.warning(f"Ollama connect failed: {e}")
            self.update_status(ProviderStatus.ERROR)
            return False

    async def disconnect(self) -> None:  # type: ignore[override]
        if self._client:
            await self._client.aclose()
        self.update_status(ProviderStatus.DISCONNECTED)

    async def get_models(self) -> List[ModelInfo]:  # type: ignore[override]
        if self._models_cache is not None:
            return self._models_cache
        if not self._client:
            await self.connect()
        try:
            assert self._client
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models: List[ModelInfo] = []
            for m in data.get("models", []):
                models.append(ModelInfo(
                    id=m.get("name"),
                    name=m.get("name"),
                    description=m.get("details", {}).get("format", "Local model"),
                    context_length=m.get("details", {}).get("context_length"),
                ))
            self._models_cache = models
            return models
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return []

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:  # type: ignore[override]
        # Non-streaming single response using /api/chat
        if not self._client:
            await self.connect()
        assert self._client
        payload = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "options": {
                "temperature": request.temperature,
                # Additional parameters can go here
            },
            "stream": False,
        }
        started = time.time()
        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = ""
        # Ollama returns message(s) in a different structure; unify
        message_obj = data.get("message") or {}
        content = message_obj.get("content", "")
        choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        return ChatCompletionResponse(
            id=data.get("id", "ollama-chat"),
            created=int(started),
            model=request.model,
            choices=[choice],
            usage=None,
        )

    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionChunk, None]:  # type: ignore[override]
        if not self._client:
            await self.connect()
        assert self._client
        payload = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "options": {"temperature": request.temperature},
            "stream": True,
        }
        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = httpx.Response(200, text=line).json()
                except Exception:
                    continue
                msg = obj.get("message", {})
                content_part = msg.get("content", "")
                if not content_part:
                    continue
                yield ChatCompletionChunk(
                    id=obj.get("id", "ollama-stream"),
                    created=int(time.time()),
                    model=request.model,
                    choices=[{"delta": {"content": content_part}, "index": 0, "finish_reason": None}],
                )

# Convenience factory
async def ensure_default_ollama(model_hint: str = "mistral") -> Optional[OllamaProvider]:
    provider = OllamaProvider()
    ok = await provider.connect()
    if not ok:
        return None
    models = await provider.get_models()
    if models and not any(m.id == model_hint for m in models):
        # Just proceed; caller can handle missing exact model
        pass
    return provider
