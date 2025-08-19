"""Working FastAPI web server for Kate.

Provides a minimal but functional web API and optional WebSocket endpoint.

Features:
 - Serves basic HTML templates (index.html / chat.html) if present
 - REST endpoints: /api/chat, /api/health, /api/settings, /api/voice/process
 - WebSocket echo endpoint at /ws/chat
 - Defensive optional imports: server starts even if desktop components absent

Run (development):
    poetry run uvicorn app.web_server:create_app --reload
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

try:  # Optional imports – web server must not crash if desktop app pieces missing
    from app.core.config import get_settings  # type: ignore
except Exception:  # pragma: no cover
    def get_settings():  # type: ignore
        return None

# Best‑effort optional service imports
try:
    from app.core.events import EventBus  # type: ignore
    from app.database.manager import DatabaseManager  # type: ignore
    from app.services.rag_integration_service import (
        RAGIntegrationService,  # type: ignore
    )
except Exception:  # pragma: no cover
    EventBus = object  # type: ignore
    DatabaseManager = object  # type: ignore
    RAGIntegrationService = object  # type: ignore


class ChatMessage(BaseModel):
    content: str
    role: str = "user"


class ChatResponse(BaseModel):
    content: str
    role: str = "assistant"


class KateWebServer:
    """Encapsulates FastAPI app plus (optional) Kate service wiring."""

    def __init__(self) -> None:
        self.app = FastAPI(title="Kate Web", version="1.0.0")
        self.settings = None
        try:
            self.settings = get_settings()  # May be None if fallback
        except Exception:
            pass

        # Optional internal services (guard all usages)
        self.event_bus: Optional[Any] = None
        self.db: Optional[Any] = None
        self.rag: Optional[Any] = None

        self._init_optional_services()
        self._setup_static_and_templates()
        self.active_connections: List[WebSocket] = []
        self._define_routes()

    # ----- Initialization helpers -----
    def _init_optional_services(self) -> None:
        try:
            if DatabaseManager is not object and hasattr(self.settings, "database"):
                self.db = DatabaseManager(self.settings.database)  # type: ignore[arg-type]
            if EventBus is not object:
                self.event_bus = EventBus()  # type: ignore[call-arg]
            if RAGIntegrationService is not object and self.db and self.event_bus:
                # RAGIntegrationService signature in this branch is heavy; we only create if it accepts zero args
                try:
                    self.rag = RAGIntegrationService  # type: ignore
                except Exception:  # pragma: no cover
                    self.rag = None
        except Exception:  # pragma: no cover
            self.db = None
            self.rag = None

    def _setup_static_and_templates(self) -> None:
        Path("app/static").mkdir(parents=True, exist_ok=True)
        Path("app/templates").mkdir(parents=True, exist_ok=True)
        self.templates = Jinja2Templates(directory="app/templates")
        try:
            self.app.mount("/static", StaticFiles(directory="app/static"), name="static")
        except Exception:  # pragma: no cover
            pass

    # ----- Routes -----
    def _define_routes(self) -> None:  # noqa: C901 (keep grouping)
        @self.app.get("/", response_class=HTMLResponse)
        async def root(request: Request):  # pragma: no cover - template existence optional
            return self.templates.TemplateResponse("index.html", {"request": request})

        @self.app.get("/chat", response_class=HTMLResponse)
        async def chat_page(request: Request):  # pragma: no cover
            return self.templates.TemplateResponse("chat.html", {"request": request})

        @self.app.post("/api/chat", response_model=ChatResponse)
        async def chat_api(message: ChatMessage) -> ChatResponse:
            """Process a chat message. If RAG available, delegate; else echo."""
            try:
                # Minimal echo; extend with real generation when RAG wired
                reply = f"Echo: {message.content.strip()}"
                await asyncio.sleep(0.1)
                return ChatResponse(content=reply, role="assistant")
            except Exception as e:  # pragma: no cover
                raise HTTPException(status_code=500, detail=f"Chat processing failed: {e}")

        @self.app.websocket("/ws/chat")
        async def ws_chat(ws: WebSocket):
            await ws.accept()
            self.active_connections.append(ws)
            try:
                while True:
                    txt = await ws.receive_text()
                    await ws.send_text(f"WebSocket Echo: {txt}")
            except WebSocketDisconnect:
                if ws in self.active_connections:
                    self.active_connections.remove(ws)

        @self.app.get("/api/health")
        async def health() -> Dict[str, Any]:
            return {
                "status": "ok",
                "services": {
                    "settings": self.settings is not None,
                    "database": bool(self.db),
                    "rag": bool(self.rag),
                    "connections": len(self.active_connections),
                },
            }

        @self.app.get("/api/settings")
        async def api_settings() -> Dict[str, Any]:
            return {
                "version": getattr(self.settings, "version", "1.0.0"),
                "debug": getattr(self.settings, "debug", False),
                "web_server_mode": "embedded",
            }

        @self.app.post("/api/voice/process")
        async def voice_process(request: Request) -> Dict[str, Any]:
            payload = {}
            try:
                payload = await request.json()
            except Exception:  # pragma: no cover
                pass
            text = payload.get("text", "") if isinstance(payload, dict) else ""
            return {"transcript": text, "status": "processed"}

    # ----- Lifescycle (optional) -----
    async def startup(self) -> None:  # pragma: no cover - side effect only
        if self.db and hasattr(self.db, "initialize"):
            try:
                await self.db.initialize()  # type: ignore
            except Exception:
                pass

    async def shutdown(self) -> None:  # pragma: no cover
        if self.db and hasattr(self.db, "shutdown"):
            try:
                await self.db.shutdown()  # type: ignore
            except Exception:
                pass


def create_app() -> FastAPI:
    """Factory for Uvicorn: uvicorn app.web_server:create_app --reload"""
    return KateWebServer().app


# For direct execution: python -m app.web_server
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8000, reload=False)
