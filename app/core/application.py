"""Core Kate application framework with PySide6 integration (clean version)."""
import asyncio
from typing import Any, List, Optional, cast

from loguru import logger
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QCloseEvent

from ..database.manager import DatabaseManager
from ..providers.base import ModelInfo
from ..providers.ollama_provider import OllamaProvider
from ..services.assistant_service import AssistantService
from ..services.real_voice_chat_service import RealVoiceChatService
from ..services.search_service import SearchService
from ..services.update_manager import UpdateManager
from ..themes.manager import (
    ThemeManager,
    cleanup_theme_manager,
    initialize_theme_manager,
)
from ..ui.main_window import MainWindow
from .config import get_settings
from .events import (
    ApplicationShutdownEvent,
    ApplicationStartedEvent,
    EventBus,
)


class KateApplication(QObject):
    """Core Kate application managing lifecycle and services."""

    startup_complete = Signal()
    shutdown_started = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._initialize_attributes()

    def _initialize_attributes(self) -> None:
        """Initialize instance attributes (split out to avoid indentation patch issues)."""
        # Core settings & services
        self.settings = get_settings()
        self.event_bus: EventBus = EventBus()
        self.database_manager: Optional[DatabaseManager] = None
        self.theme_manager: Optional[ThemeManager] = None
        self.search_service: Optional[SearchService] = None
        self.update_manager: Optional[UpdateManager] = None
        self.assistant_service: Optional[AssistantService] = None
        self.voice_service: Optional[RealVoiceChatService] = None
        self.main_window: Optional[MainWindow] = None

        # LLM provider (Ollama) integration attributes (populated later)
        self.ollama_provider: Optional[OllamaProvider] = None
        self.available_models: List[ModelInfo] = []
        self.selected_model: Optional[str] = None

        # Lifecycle flags & shutdown tracking
        self._initialized = False
        self._shutting_down = False
        self._shutdown_complete = False
        self._shutdown_task: Optional[asyncio.Task[None]] = None
        self._shutdown_event = asyncio.Event()

        # Logging & timers
        self.logger = logger.bind(component="KateApplication")
        self._auto_save_timer = QTimer()
        self._auto_save_timer.timeout.connect(lambda: asyncio.create_task(self._auto_save()))



    # ----- Lifecycle -----
    async def startup(self) -> None:
        if self._initialized:
            self.logger.warning("Application already initialized")
            return
        try:
            self.logger.info("Starting Kate application...")
            await self._initialize_database()
            await self._initialize_themes()
            await self._initialize_ui()
            await self._start_services()
            self._setup_event_handlers()
            if self.settings.auto_save_interval > 0:
                self._auto_save_timer.start(self.settings.auto_save_interval * 1000)
            self._initialized = True
            await self.event_bus.emit_async(ApplicationStartedEvent())
            self.startup_complete.emit()
            self.logger.info("Kate application started successfully")
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Failed to start application: {e}")
            raise

    async def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self.shutdown_started.emit()
        try:
            self.logger.info("Shutting down Kate application...")
            await self.event_bus.emit_async(ApplicationShutdownEvent())
            self._auto_save_timer.stop()
            await self._auto_save()
            if self.main_window:
                self.main_window.close()
            if self.update_manager:
                await self.update_manager.cleanup()
            if self.voice_service and self.voice_service.is_active():
                self.logger.info("Stopping active voice session...")
                # Voice service will stop automatically when app shuts down
            if self.search_service:
                await self.search_service.cleanup()
            if self.theme_manager:
                cleanup_theme_manager()
            if self.database_manager:
                await self.database_manager.shutdown()
            self.event_bus.clear_handlers()
            self.logger.info("Kate application shut down successfully")
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            self._shutdown_complete = True
            self._shutdown_event.set()

    async def restart(self) -> None:
        self.logger.info("Restarting Kate application...")
        await self.shutdown()
        await self.startup()

    # ----- Initialization helpers -----
    async def _initialize_database(self) -> None:
        self.logger.info("Initializing database...")
        self.database_manager = DatabaseManager(self.settings.database)
        await self.database_manager.initialize()
        await self.database_manager.migrate()
        self.logger.info("Database initialized successfully")

    async def _initialize_themes(self) -> None:
        self.logger.info("Initializing theme system...")
        themes_dir = self.settings.config_dir / "themes"
        self.theme_manager = initialize_theme_manager(event_bus=self.event_bus, themes_dir=themes_dir)
        current_theme = self.settings.ui.theme
        if not self.theme_manager.apply_theme(current_theme):
            self.logger.warning(f"Failed to apply theme '{current_theme}', using default")
            self.theme_manager.set_default_theme()
        self.logger.info("Theme system initialized successfully")

    async def _initialize_ui(self) -> None:
        self.logger.info("Initializing UI...")
        self.main_window = MainWindow(app=self, settings=self.settings, event_bus=self.event_bus)
        self.main_window.closeEvent = self._on_window_close  # type: ignore[method-assign]
        self.main_window.show()
        self._restore_window_geometry()
        self.logger.info("UI initialized successfully")

    async def _start_services(self) -> None:
        self.logger.info("Starting application services...")
        # Database manager must be initialized at this point
        assert self.database_manager is not None
        self.search_service = SearchService(
            database_manager=self.database_manager,
            event_bus=self.event_bus,
        )
        await self.search_service.initialize()
        # Initialize assistant service (loads assistant definitions / fallback)
        self.assistant_service = AssistantService()
        self.update_manager = UpdateManager(
            settings=self.settings,
            event_bus=self.event_bus,
        )
        
        # Initialize voice chat service (TTS/STT)
        self.voice_service = RealVoiceChatService(voice_settings=self.settings.voice)
        capabilities = self.voice_service.get_capabilities()
        self.logger.info(f"Voice service initialized - capabilities: {capabilities}")
        
        self.logger.info("Application services started")

        # Attempt to initialize local Ollama provider (non-fatal if unavailable)
        await self._initialize_ollama_provider()

    async def _initialize_ollama_provider(self) -> None:
        """Initialize connection to local Ollama server and populate models.

        This is best-effort: failures are logged but do not prevent UI startup.
        """
        try:
            self.logger.info("Connecting to local Ollama provider (if available)...")
            provider = OllamaProvider()
            ok = await provider.connect()
            if not ok:
                self.logger.warning("Ollama provider not reachable; continuing without local LLM")
                return
            models = await provider.get_models()
            self.ollama_provider = provider
            self.available_models = models
            # Choose default model: prefer one matching fallback assistant model or first
            preferred = "mistral"
            selected = None
            for m in models:
                if m.id.startswith(preferred):
                    selected = m.id
                    break
            if not selected and models:
                selected = models[0].id
            self.selected_model = selected
            self.logger.info(f"Ollama connected; {len(models)} models available; selected={self.selected_model}")
            # Update status bar if UI already present
            if self.main_window and hasattr(self.main_window, 'status_bar'):
                try:
                    self.main_window.status_bar.set_connection_status(True)
                    if self.selected_model:
                        self.main_window.status_bar.set_provider_info('Ollama', self.selected_model)
                    else:
                        self.main_window.status_bar.set_provider_info('Ollama')
                except Exception:
                    pass
        except Exception as e:  # pragma: no cover
            self.logger.warning(f"Failed to initialize Ollama provider: {e}")

    async def refresh_ollama_models(self) -> None:
        """Refresh the list of available Ollama models (if provider active)."""
        if not self.ollama_provider:
            return
        try:
            models = await self.ollama_provider.get_models()
            self.available_models = models
            if self.selected_model and not any(m.id == self.selected_model for m in models):
                # Previously selected model missing; choose first
                self.selected_model = models[0].id if models else None
            if self.main_window and hasattr(self.main_window, 'status_bar') and self.selected_model:
                self.main_window.status_bar.set_provider_info('Ollama', self.selected_model)
            self.logger.info(f"Refreshed Ollama models; count={len(models)}")
        except Exception as e:  # pragma: no cover
            self.logger.warning(f"Failed to refresh Ollama models: {e}")

    # ----- Event handling -----
    def _setup_event_handlers(self) -> None:
        # Cast to Any due to mypy's invariance on Callable parameter types
        self.event_bus.subscribe_async(ApplicationShutdownEvent, cast(Any, self._on_shutdown_event))
        self.logger.debug("Event handlers configured")

    async def _on_shutdown_event(self, event: ApplicationShutdownEvent) -> None:
        await self._save_application_state()

    def _on_window_close(self, event: QCloseEvent) -> None:
        self.logger.info("Main window close requested")
        self._save_window_geometry()
        if not self._shutdown_task:
            self._shutdown_task = asyncio.create_task(self.shutdown())
        event.accept()

    # ----- Persistence helpers -----
    def _restore_window_geometry(self) -> None:
        if not self.main_window:
            return
        geometry = self.settings.window
        if geometry.width and geometry.height:
            self.main_window.resize(geometry.width, geometry.height)
        if geometry.x is not None and geometry.y is not None:
            self.main_window.move(geometry.x, geometry.y)
        if geometry.maximized:
            self.main_window.showMaximized()

    def _save_window_geometry(self) -> None:
        if not self.main_window:
            return
        geometry = self.main_window.geometry()
        self.settings.window.width = geometry.width()
        self.settings.window.height = geometry.height()
        self.settings.window.x = geometry.x()
        self.settings.window.y = geometry.y()
        self.settings.window.maximized = self.main_window.isMaximized()

    async def _auto_save(self) -> None:
        try:
            await self._save_application_state()
            self.logger.debug("Auto-save completed")
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Auto-save failed: {e}")

    async def _save_application_state(self) -> None:
        self._save_window_geometry()

    # ----- Accessors -----
    def get_main_window(self) -> Optional[MainWindow]:
        return self.main_window

    def is_initialized(self) -> bool:
        return self._initialized

    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def is_shutdown_complete(self) -> bool:
        return self._shutdown_complete

    async def wait_for_shutdown(self) -> None:
        await self._shutdown_event.wait()

    def get_theme_manager(self) -> Optional[ThemeManager]:
        return self.theme_manager
    
    def get_voice_service(self) -> Optional[RealVoiceChatService]:
        """Get the voice chat service for TTS/STT functionality."""
        return self.voice_service