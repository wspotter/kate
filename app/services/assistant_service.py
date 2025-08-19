"""Assistant service for loading and validating assistant definitions.

Centralises assistant config concerns (path resolution, validation, reload)
so UI components stay thin. Designed to support future features:
 - Hot reload via file watcher
 - Remote / user-defined assistant registries
 - Per-assistant parameter presets (temperature etc.)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

try:  # pydantic already in dependencies; fallback guards tests if import fails early
    from pydantic import (  # type: ignore
        BaseModel,
        Field,
        field_validator,  # type: ignore  # v2 API
    )
except Exception:  # pragma: no cover - should not occur in normal runtime
    class _FallbackBase:  # minimal placeholder
        pass
    class _FallbackModel(_FallbackBase):  # type: ignore
        def __init__(self, **data):  # accept anything
            for k, v in data.items():
                setattr(self, k, v)
        def dict(self):  # mimic pydantic API subset
            return self.__dict__
    BaseModel = _FallbackModel  # type: ignore
    def Field(*_a, **_k):  # type: ignore
        return None
    def field_validator(*_a, **_k):  # type: ignore
        def _wrap(fn):  # type: ignore
            return fn  # type: ignore
        return _wrap  # type: ignore


class AssistantDefinition(BaseModel):  # type: ignore[misc]
    name: str = Field(..., description="Display name")  # type: ignore[assignment]
    description: str = Field(..., description="Short description of role")  # type: ignore[assignment]
    provider: str = Field(..., description="Provider key (e.g. ollama)")  # type: ignore[assignment]
    model: str = Field(..., description="Model identifier or prefix")  # type: ignore[assignment]
    system_prompt: Optional[str] = Field(None, description="Optional explicit system prompt")  # type: ignore[assignment]
    avatar: Optional[str] = Field("🤖", description="Emoji or small text avatar")  # type: ignore[assignment]

    @field_validator("name", "description", "provider", "model")  # type: ignore[override]
    @classmethod
    def not_empty(cls, v: str) -> str:  # type: ignore[override]
        if not v or not v.strip():  # pragma: no cover - simple guard
            raise ValueError("must not be empty")
        return v


class AssistantService:
    """Service responsible for loading assistant definitions from JSON.

    Contract:
      - load(path) loads JSON object mapping id -> definition
      - validation errors logged; invalid assistants skipped, at least one fallback retained
      - get_assistants returns dict of validated assistants
      - reload() re-reads from last path
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path: Optional[Path] = Path(config_path) if config_path else None
        self._assistants: Dict[str, Dict[str, Any]] = {}
        self._logger = logger.bind(component="AssistantService")

        if self._config_path and self._config_path.is_file():
            self._load_from_path(self._config_path)
        else:
            # Will lazily attempt default path resolution on first access
            pass

    # ---------------------------- Public API ---------------------------- #
    def configure_path(self, path: Path) -> None:
        self._config_path = path

    def get_config_path(self) -> Optional[Path]:
        """Return current assistants config path (resolved) if known."""
        return self._config_path

    def get_assistants(self) -> Dict[str, Dict[str, Any]]:
        if not self._assistants:
            self._ensure_loaded()
        return self._assistants

    def reload(self) -> Dict[str, Dict[str, Any]]:
        self._assistants = {}
        self._ensure_loaded(force=True)
        if not self._assistants:
            self._logger.warning("Assistant reload produced zero assistants; fallback applied")
        return self._assistants

    # --------------------------- Internal Logic ------------------------- #
    def _ensure_loaded(self, force: bool = False) -> None:
        if self._assistants and not force:
            return
        path = self._resolve_default_path() if not self._config_path else self._config_path
        if path and path.is_file():
            self._load_from_path(path)
        if not self._assistants:
            self._apply_fallback()

    def _resolve_default_path(self) -> Optional[Path]:
        # Expect structure app/services/ -> ascend to app/ then config/assistants.json
        here = Path(__file__).resolve()
        app_dir = here.parent.parent  # app/
        candidate = app_dir / "config" / "assistants.json"
        if candidate.is_file():
            self._config_path = candidate
            return candidate
        self._logger.debug(f"Assistant config not found at default path: {candidate}")
        return None

    def _load_from_path(self, path: Path) -> None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("assistant config root must be an object mapping id -> definition")
            valid: Dict[str, Dict[str, Any]] = {}
            for key, value in raw.items():  # type: ignore[assignment]
                if not isinstance(value, dict):
                    self._logger.warning(f"Assistant '{key}' skipped (not an object)")
                    continue
                try:
                    model_obj = AssistantDefinition(**value)
                    # pydantic v2: model_dump; fallback stub supplies dict()
                    dump_fn = getattr(model_obj, "model_dump", None)
                    data_dict = dump_fn() if callable(dump_fn) else model_obj.__dict__  # type: ignore[attr-defined]
                    valid[key] = data_dict  # type: ignore[assignment]
                except Exception as ve:  # validation error
                    self._logger.warning(f"Assistant '{key}' invalid: {ve}")
            if valid:
                self._assistants = valid
                self._logger.info(f"Loaded {len(valid)} assistants from {path}")
            else:
                self._logger.warning("No valid assistants after validation; applying fallback")
        except Exception as e:
            self._logger.warning(f"Failed reading assistants file {path}: {e}")
        if not self._assistants:
            self._apply_fallback()

    def _apply_fallback(self) -> None:
        self._assistants = {
            "general": {
                "name": "General Assistant",
                "description": "Fallback general-purpose assistant.",
                "system_prompt": "You are a helpful, concise assistant.",
                "avatar": "🤖",
                "provider": "ollama",
                "model": "mistral",
            }
        }
        self._logger.info("Using fallback assistant definition")


__all__ = [
    "AssistantService",
    "AssistantDefinition",
]
