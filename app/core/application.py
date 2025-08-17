"""
Core Kate application framework with PySide6 integration.
"""
import asyncio
import sys
from typing import Optional
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QObject, Signal
from loguru import logger

from .config import AppSettings, get_settings
from .events import EventBus, ApplicationStartedEvent, ApplicationShutdownEvent
from ..database.manager import DatabaseManager
from ..ui.main_window import MainWindow
from ..themes.manager import ThemeManager, initialize_theme_manager, cleanup_theme_manager
from ..services.search_service import SearchService
from ..services.update_manager import UpdateManager


class KateApplication(QObject):
    """
    Core Kate application managing lifecycle and services.
    
    This class coordinates all major application components:
    - Configuration management
    - Database connectivity
    - Event system
    - UI management
    - Service lifecycle
    """
    
    # Qt signals for inter-component communication
    startup_complete = Signal()
    shutdown_started = Signal()
    
    def __init__(self):
        super().__init__()
        
        # Core services
        self.settings: AppSettings = get_settings()
        self.event_bus: EventBus = EventBus()
        self.database_manager: Optional[DatabaseManager] = None
        self.theme_manager: Optional[ThemeManager] = None
        self.search_service: Optional[SearchService] = None
        self.update_manager: Optional[UpdateManager] = None
        self.main_window: Optional[MainWindow] = None
        
        # Application state
        self._initialized = False
        self._shutting_down = False
        
        # Setup logging
        self.logger = logger.bind(component="KateApplication")
        
        # Auto-save timer
        self._auto_save_timer = QTimer()
        self._auto_save_timer.timeout.connect(self._auto_save)
        
    async def startup(self) -> None:
        """
        Initialize and start all application services.
        
        This method must be called before the application can be used.
        """
        if self._initialized:
            self.logger.warning("Application already initialized")
            return
            
        try:
            self.logger.info("Starting Kate application...")
            
            # Initialize database
            await self._initialize_database()
            
            # Initialize theme system
            await self._initialize_themes()
            
            # Initialize UI
            await self._initialize_ui()
            
            # Start services
            await self._start_services()
            
            # Setup event handlers
            self._setup_event_handlers()
            
            # Start auto-save timer
            if self.settings.auto_save_interval > 0:
                self._auto_save_timer.start(self.settings.auto_save_interval * 1000)
            
            self._initialized = True
            
            # Emit startup event
            await self.event_bus.emit_async(ApplicationStartedEvent())
            self.startup_complete.emit()
            
            self.logger.info("Kate application started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start application: {e}")
            raise
            
    async def shutdown(self) -> None:
        """
        Gracefully shutdown all application services.
        """
        if self._shutting_down:
            return
            
        self._shutting_down = True
        self.shutdown_started.emit()
        
        try:
            self.logger.info("Shutting down Kate application...")
            
            # Emit shutdown event
            await self.event_bus.emit_async(ApplicationShutdownEvent())
            
            # Stop auto-save timer
            self._auto_save_timer.stop()
            
            # Perform final save
            await self._auto_save()
            
            # Shutdown UI
            if self.main_window:
                self.main_window.close()
            
            # Shutdown services
            if self.update_manager:
                await self.update_manager.cleanup()
                
            if self.search_service:
                await self.search_service.cleanup()
            
            # Shutdown theme manager
            if self.theme_manager:
                cleanup_theme_manager()
            
            # Shutdown database
            if self.database_manager:
                await self.database_manager.shutdown()
            
            # Clear event handlers
            self.event_bus.clear_handlers()
            
            self.logger.info("Kate application shut down successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            
    async def restart(self) -> None:
        """
        Restart the application.
        """
        self.logger.info("Restarting Kate application...")
        await self.shutdown()
        await self.startup()
        
    async def _initialize_database(self) -> None:
        """Initialize the database connection and schema."""
        self.logger.info("Initializing database...")
        
        self.database_manager = DatabaseManager(self.settings.database)
        await self.database_manager.initialize()
        
        # Run migrations if needed
        await self.database_manager.migrate()
        
        self.logger.info("Database initialized successfully")
        
    async def _initialize_themes(self) -> None:
        """Initialize the theme system."""
        self.logger.info("Initializing theme system...")
        
        # Create themes directory
        themes_dir = self.settings.config_dir / "themes"
        
        # Initialize theme manager
        self.theme_manager = initialize_theme_manager(
            event_bus=self.event_bus,
            themes_dir=themes_dir
        )
        
        # Apply the configured theme
        current_theme = self.settings.ui.theme
        if not self.theme_manager.apply_theme(current_theme):
            # Fall back to default theme if configured theme fails
            self.logger.warning(f"Failed to apply theme '{current_theme}', using default")
            self.theme_manager.set_default_theme()
        
        self.logger.info("Theme system initialized successfully")
        
    async def _initialize_ui(self) -> None:
        """Initialize the main UI window."""
        self.logger.info("Initializing UI...")
        
        # Create main window with the correct parameters that the constructor expects
        self.main_window = MainWindow(
            app=self,
            settings=self.settings,
            event_bus=self.event_bus
        )
        
        # Connect window close event to shutdown
        self.main_window.closeEvent = self._on_window_close
        
        # Show the window
        self.main_window.show()
        
        # Restore window geometry
        self._restore_window_geometry()
        
        self.logger.info("UI initialized successfully")
        
    async def _start_services(self) -> None:
        """Start all application services."""
        self.logger.info("Starting application services...")
        
        # Initialize search service
        self.search_service = SearchService(
            database_manager=self.database_manager,
            event_bus=self.event_bus
        )
        await self.search_service.initialize()
        
        # Initialize update manager with correct constructor parameters
        self.update_manager = UpdateManager(
            settings=self.settings,
            event_bus=self.event_bus
        )
        
        self.logger.info("Application services started")
        
    def _setup_event_handlers(self) -> None:
        """Setup global event handlers."""
        # Subscribe to application events
        self.event_bus.subscribe_async(ApplicationShutdownEvent, self._on_shutdown_event)
        
        self.logger.debug("Event handlers configured")
        
    async def _on_shutdown_event(self, event: ApplicationShutdownEvent) -> None:
        """Handle application shutdown event."""
        # Save current state before shutdown
        await self._save_application_state()
        
    def _on_window_close(self, event) -> None:
        """Handle main window close event."""
        self.logger.info("Main window close requested")
        
        # Save window geometry
        self._save_window_geometry()
        
        # Start async shutdown
        asyncio.create_task(self.shutdown())
        
        event.accept()
        
    def _restore_window_geometry(self) -> None:
        """Restore window geometry from settings."""
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
        """Save current window geometry to settings."""
        if not self.main_window:
            return
            
        geometry = self.main_window.geometry()
        
        self.settings.window.width = geometry.width()
        self.settings.window.height = geometry.height()
        self.settings.window.x = geometry.x()
        self.settings.window.y = geometry.y()
        self.settings.window.maximized = self.main_window.isMaximized()
        
    async def _auto_save(self) -> None:
        """Perform automatic save of application state."""
        try:
            await self._save_application_state()
            self.logger.debug("Auto-save completed")
        except Exception as e:
            self.logger.error(f"Auto-save failed: {e}")
            
    async def _save_application_state(self) -> None:
        """Save current application state."""
        # Save window geometry
        self._save_window_geometry()
        
        # Additional state saving will be implemented as features are added
        
    def get_main_window(self) -> Optional[MainWindow]:
        """Get the main application window."""
        return self.main_window
        
    def is_initialized(self) -> bool:
        """Check if the application is initialized."""
        return self._initialized
        
    def is_shutting_down(self) -> bool:
        """Check if the application is shutting down."""
        return self._shutting_down
        
    def get_theme_manager(self) -> Optional[ThemeManager]:
        """Get the theme manager."""
        return self.theme_manager