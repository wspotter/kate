"""
Main window for Kate LLM Client with 3-column layout.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, cast

from loguru import logger
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (  # noqa: F401 (reserved for future window icon usage)
    QAction,
    QIcon,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QWidget,
)

from ..core.config import AppSettings
from ..core.events import EventBus
from .components.assistant_panel import AssistantPanel
from .components.chat_area import ChatArea
from .components.conversation_sidebar import ConversationSidebar
from .components.evaluation_dashboard import EvaluationDashboard
from .components.settings_window import SettingsWindow
from .components.status_bar import StatusBar

if TYPE_CHECKING:
    from ..core.application import KateApplication
else:
    KateApplication = Any


class MainWindow(QMainWindow):
    """
    Main application window with 3-column layout:
    - Left: Conversation sidebar
    - Center: Chat area
    - Right: Assistant panel
    """
    
    # Signals
    closing = Signal()
    
    # Attribute type declarations (for mypy clarity)
    app: KateApplication
    settings: AppSettings
    event_bus: EventBus
    logger: Any
    evaluation_service: Any
    evaluation_dashboard: Optional[EvaluationDashboard]
    central_widget: QWidget
    main_layout: QHBoxLayout
    main_splitter: QSplitter
    conversation_sidebar: ConversationSidebar
    chat_area: ChatArea
    assistant_panel: AssistantPanel
    status_bar: StatusBar
    settings_window: Optional[SettingsWindow]

    def __init__(self, app: KateApplication, settings: AppSettings, event_bus: EventBus):
        super().__init__()
        self.app = app
        self.settings = settings
        self.event_bus = event_bus
        self.logger = logger.bind(component="MainWindow")

        # Initialize evaluation service (assigned later to concrete service)
        self.evaluation_service = None  # type: ignore[assignment]
        self.evaluation_dashboard = None
        self.settings_window = None

        # Initialize UI components
        self._setup_ui()
        self._setup_menu()
        self._setup_layout()
        self._connect_signals()
        self._setup_evaluation_service()

        # Apply theme
        self._apply_theme()

        self.logger.info("Main window initialized with 3-column layout and evaluation integration")

        # Add a timer to check widget visibility after a short delay
        QTimer.singleShot(1000, self._verify_widget_display)
        
    def _setup_ui(self) -> None:
        """Set up the main window UI."""
        self.logger.info("Setting up main window UI...")
        # Window properties
        self.setWindowTitle("Kate - LLM Desktop Client")
        self.setMinimumSize(1100, 760)
        self.resize(1480, 900)
        self.logger.info("Window properties set - title, minimum size, and resize")

        # Set window icon (placeholder)
        # self.setWindowIcon(QIcon(":/icons/kate.png"))  # TODO: Add icon
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.logger.info("Central widget created and set")

        # Create main layout
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.logger.info("Main horizontal layout created with zero margins and spacing")
    
    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        self.logger.info("Setting up menu bar...")

        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        new_conversation_action = QAction("&New Conversation", self)
        new_conversation_action.setShortcut("Ctrl+N")
        new_conversation_action.triggered.connect(self.create_new_conversation)
        file_menu.addAction(new_conversation_action)
        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        settings_action = QAction("&Preferences...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._open_settings)
        settings_menu.addAction(settings_action)

        # View menu
        view_menu = menubar.addMenu("&View")
        toggle_sidebar_action = QAction("Toggle &Sidebar", self)
        toggle_sidebar_action.setShortcut("Ctrl+1")
        toggle_sidebar_action.triggered.connect(self.toggle_conversation_sidebar)
        view_menu.addAction(toggle_sidebar_action)
        toggle_assistant_action = QAction("Toggle &Assistant Panel", self)
        toggle_assistant_action.setShortcut("Ctrl+2")
        toggle_assistant_action.triggered.connect(self.toggle_assistant_panel)
        view_menu.addAction(toggle_assistant_action)
        view_menu.addSeparator()
        eval_dashboard_action = QAction("&Evaluation Dashboard", self)
        eval_dashboard_action.setShortcut("Ctrl+E")
        eval_dashboard_action.triggered.connect(self.show_evaluation_dashboard)
        view_menu.addAction(eval_dashboard_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About Kate", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        self.logger.info("Menu bar created with File, Settings, View, and Help menus")
        
    def _setup_layout(self) -> None:
        """Set up the 3-column layout."""
        self.logger.info("Setting up 3-column layout...")
        # Create splitter
        try:
            self.main_splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]
        except Exception:
            self.main_splitter = QSplitter()
        self.main_splitter.setChildrenCollapsible(False)
        # Left panel
        self.conversation_sidebar = ConversationSidebar(self.event_bus)
        self.conversation_sidebar.setMinimumWidth(220)
        self.conversation_sidebar.setMaximumWidth(420)
        # Center panel
        self.chat_area = ChatArea(self.event_bus)
        # Right panel
        shared_assistant_service = getattr(self.app, 'assistant_service', None)
        self.assistant_panel = AssistantPanel(self.event_bus, assistant_service=shared_assistant_service)
        self.assistant_panel.setMinimumWidth(300)
        try:
            self.assistant_panel.setMaximumWidth(16777215)
        except Exception:
            pass
        # Assemble splitter
        self.main_splitter.addWidget(self.conversation_sidebar)
        self.main_splitter.addWidget(self.chat_area)
        self.main_splitter.addWidget(self.assistant_panel)
        try:
            self.main_splitter.setStretchFactor(0, 0)
            self.main_splitter.setStretchFactor(1, 1)
            self.main_splitter.setStretchFactor(2, 0)
        except Exception:
            pass
        self._initial_sizes_applied = False
        self.main_layout.addWidget(self.main_splitter)
        # Status bar
        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)
        self.logger.info("3-column layout ready")

    def showEvent(self, event):  # type: ignore[override]
        """On first show, apply proportional splitter sizes (approx 22/56/22)."""
        super().showEvent(event)
        if not getattr(self, "_initial_sizes_applied", False):
            try:
                total = max(self.width(), 1)
                left = max(self.conversation_sidebar.minimumWidth(), min(int(total * 0.22), 420))
                right = max(self.assistant_panel.minimumWidth(), min(int(total * 0.22), 520))
                center = max(400, total - (left + right))
                self.main_splitter.setSizes([left, center, right])
                self.logger.info(f"Applied initial splitter sizes left={left} center={center} right={right}")
            except Exception as e:
                self.logger.debug(f"Failed to apply initial dynamic splitter sizes: {e}")
            self._initial_sizes_applied = True
        
    def _connect_signals(self) -> None:
        """Connect UI signals."""
        # Connect conversation sidebar signals
        self.conversation_sidebar.conversation_selected.connect(
            self.chat_area.load_conversation
        )
        
        # Connect chat area signals
        self.chat_area.message_sent.connect(
            self._handle_message_sent
        )
        
        # Connect assistant panel signals
        self.assistant_panel.assistant_changed.connect(
            self._handle_assistant_changed
        )
        self.assistant_panel.model_settings_changed.connect(
            self._handle_model_settings_changed
        )
        self.assistant_panel.evaluation_details_requested.connect(
            self._handle_evaluation_details_request
        )
        
        # Connect chat area evaluation signals
        self.chat_area.evaluation_received.connect(
            self._handle_evaluation_received
        )
        
    def _apply_theme(self) -> None:
        """Apply the current theme to the window."""
        # Get theme from theme manager
        if hasattr(self.app, 'theme_manager') and self.app.theme_manager:
            current_theme = self.app.theme_manager.current_theme
            if current_theme:
                stylesheet = self.app.theme_manager._generate_stylesheet(current_theme)
                self.setStyleSheet(stylesheet)
            
    def _handle_message_sent(self, message: str) -> None:
        """Handle message sent from chat area."""
        self.logger.debug(f"Message sent: {message[:50]}...")
        # Message handling will be implemented when connecting to LLM providers
        
    def _handle_assistant_changed(self, assistant_id: str) -> None:
        """Handle assistant selection change."""
        self.logger.debug(f"Assistant changed to: {assistant_id}")
        # Provide assistant metadata to chat area for system prompt adaptation
        if hasattr(self.assistant_panel, 'assistants') and assistant_id in self.assistant_panel.assistants:
            data = self.assistant_panel.assistants[assistant_id]
            if hasattr(self.chat_area, 'set_assistant'):
                self.chat_area.set_assistant(assistant_id, data)
            # Attempt to switch model if assistant specifies an ollama model
            try:
                desired_model = data.get('model')
                provider_name = data.get('provider')
                if provider_name == 'ollama' and desired_model:
                    class _HasModels(Protocol):  # local Protocol to satisfy mypy
                        available_models: List[Any]
                        selected_model: str

                    app_with_models = cast(Optional[_HasModels], self.app if hasattr(self.app, 'available_models') else None)
                    if app_with_models and getattr(app_with_models, 'available_models', None):
                        for m in app_with_models.available_models:  # type: ignore[attr-defined]
                            mid = getattr(m, 'id', None)
                            if isinstance(mid, str) and isinstance(desired_model, str) and mid.startswith(desired_model):
                                try:
                                    app_with_models.selected_model = mid  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                                status_bar = getattr(getattr(self.app, 'main_window', None), 'status_bar', None)
                                if status_bar and hasattr(status_bar, 'set_provider_info'):
                                    try:
                                        status_bar.set_provider_info('Ollama', mid)
                                    except Exception:
                                        pass
                                self.logger.info(f"Switched model due to assistant selection: {mid}")
                                break
            except Exception as e:
                self.logger.debug(f"Model switch on assistant change skipped: {e}")

    def _handle_model_settings_changed(self, settings: Dict[str, Any]) -> None:
        """Propagate model parameter changes to chat area."""
        if hasattr(self.chat_area, 'set_model_settings'):
            self.chat_area.set_model_settings(settings)
        self.logger.debug("Model settings updated from assistant panel")
        
    def _handle_evaluation_received(self, evaluation_data) -> None:
        """Handle evaluation data received from chat area."""
        try:
            # Update assistant panel with evaluation metrics
            if hasattr(evaluation_data, 'to_dict'):
                # ResponseEvaluation object
                self.assistant_panel.update_evaluation(evaluation_data)
            else:
                # Already a dictionary
                # For dict-based stub evaluations, we currently skip assistant_panel.update_evaluation
                pass
                
            # Update evaluation dashboard if available
            if self.evaluation_dashboard and hasattr(evaluation_data, 'to_dict'):
                self.evaluation_dashboard.add_evaluation(evaluation_data)
                
            self.logger.debug("Evaluation data processed and distributed to UI components")
            
        except Exception as e:
            self.logger.error(f"Failed to handle evaluation data: {e}")
            
    def _handle_evaluation_details_request(self, evaluation_data) -> None:
        """Handle request to show detailed evaluation information."""
        try:
            if not self.evaluation_dashboard:
                self._create_evaluation_dashboard()
                
            # Show the evaluation dashboard
            dash = self.evaluation_dashboard
            if dash:
                dash.show()
                if hasattr(dash, 'raise_'):
                    dash.raise_()
                if hasattr(dash, 'activateWindow'):
                    dash.activateWindow()
            
            self.logger.info("Evaluation dashboard opened")
            
        except Exception as e:
            self.logger.error(f"Failed to show evaluation dashboard: {e}")
            
    def _setup_evaluation_service(self) -> None:
        """Set up the evaluation service and connect it to components."""
        try:
            # Lazy import to avoid hanging during startup
            from ..services.embedding_service import EmbeddingService
            from ..services.rag_evaluation_service import RAGEvaluationService
            
            # Create a minimal embedding service for evaluation
            # In a full implementation, this would be injected from the application
            embedding_service = EmbeddingService(  # type: ignore[arg-type]
                database_manager=None,  # TODO: inject real DatabaseManager
                event_bus=self.event_bus,
            )
            
            # Create evaluation service
            self.evaluation_service = RAGEvaluationService(embedding_service)
            
            # Connect evaluation service to components
            self.chat_area.set_evaluation_service(self.evaluation_service)
            self.assistant_panel.set_evaluation_service(self.evaluation_service)
            
            self.logger.info("Evaluation service initialized and connected")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize evaluation service: {e}")
            # Continue without evaluation service
            
    def _create_evaluation_dashboard(self) -> None:
        """Create the evaluation dashboard window."""
        try:
            if not self.evaluation_dashboard:
                self.evaluation_dashboard = EvaluationDashboard(
                    self.event_bus,
                    self.evaluation_service
                )
                
                # Set up as a separate window
                self.evaluation_dashboard.setWindowTitle("RAG Evaluation Dashboard")
                self.evaluation_dashboard.resize(1000, 700)
                
                # Connect export signal
                self.evaluation_dashboard.export_requested.connect(
                    self._handle_evaluation_export
                )
                
                self.logger.info("Evaluation dashboard created")
                
        except Exception as e:
            self.logger.error(f"Failed to create evaluation dashboard: {e}")
            
    def _handle_evaluation_export(self, file_path: str) -> None:
        """Handle evaluation data export."""
        try:
            self.update_status(f"Evaluation data exported to: {file_path}")
            self.logger.info(f"Evaluation data exported successfully to: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to handle evaluation export: {e}")
        
    def _handle_rag_context_updated(self, sources: list) -> None:
        """Handle RAG context updates."""
        self.logger.debug(f"RAG context updated with {len(sources)} sources")
        # Update assistant panel with context information
        # This could show retrieved sources or context relevance
        self.update_status(f"RAG context: {len(sources)} sources retrieved")
        
    def update_indexing_status(self, active_tasks: int, queued_tasks: int, current_task: str = "") -> None:
        """Update indexing status across UI components."""
        # Update assistant panel
        self.assistant_panel.update_indexing_status(active_tasks, queued_tasks, current_task)
        
        # Update status bar
        if active_tasks > 0 or queued_tasks > 0:
            status_parts = []
            if active_tasks > 0:
                status_parts.append(f"{active_tasks} indexing")
            if queued_tasks > 0:
                status_parts.append(f"{queued_tasks} queued")
            self.update_status("Document indexing: " + ", ".join(status_parts))
        else:
            self.update_status("Ready")
        
    def show_conversation(self, conversation_id: str) -> None:
        """Show a specific conversation."""
        self.conversation_sidebar.select_conversation(conversation_id)
        
    def create_new_conversation(self) -> None:
        """Create a new conversation."""
        self.conversation_sidebar.create_new_conversation()
        
    def closeEvent(self, event) -> None:
        """Handle window close event."""
        self.logger.info("Main window closing")
        self.closing.emit()
        
        # Cleanup evaluation resources
        self.cleanup_evaluation()
        
        # Save window state
        self._save_window_state()
        
        # Accept the close event
        event.accept()
        
    def _save_window_state(self) -> None:
        """Save the current window state."""
        # Save splitter sizes and window geometry
        # TODO: Implement settings persistence
        pass
        
    def _restore_window_state(self) -> None:
        """Restore the saved window state."""
        # Restore splitter sizes and window geometry
        # TODO: Implement settings loading
        pass
        
    def update_status(self, message: str) -> None:
        """Update the status bar message."""
        self.status_bar.showMessage(message)
        
    def set_loading(self, loading: bool) -> None:
        """Set loading state for the entire window."""
        # TODO: Implement loading indicator
        pass
        
    def toggle_assistant_panel(self) -> None:
        """Toggle the assistant panel visibility."""
        if self.assistant_panel.isVisible():
            self.assistant_panel.hide()
        else:
            self.assistant_panel.show()
            
    def toggle_conversation_sidebar(self) -> None:
        """Toggle the conversation sidebar visibility."""
        if self.conversation_sidebar.isVisible():
            self.conversation_sidebar.hide()
        else:
            self.conversation_sidebar.show()
            
    def set_rag_integration_service(self, rag_integration_service) -> None:
        """Set the RAG integration service for the chat area."""
        self.chat_area.set_rag_integration_service(rag_integration_service)
        
        # If evaluation service is available, connect it to the RAG integration
        if self.evaluation_service and hasattr(rag_integration_service, 'rag_evaluation_service'):
            rag_integration_service.rag_evaluation_service = self.evaluation_service
            
        self.logger.info("RAG integration service connected to main window")
        
    def get_chat_area(self) -> ChatArea:
        """Get the chat area component."""
        return self.chat_area
        
    def get_conversation_sidebar(self) -> ConversationSidebar:
        """Get the conversation sidebar component."""
        return self.conversation_sidebar
        
    def get_assistant_panel(self) -> AssistantPanel:
        """Get the assistant panel component."""
        return self.assistant_panel
        
    def get_evaluation_service(self):
        """Get the evaluation service."""
        return self.evaluation_service
        
    def show_evaluation_dashboard(self) -> None:
        """Show the evaluation dashboard."""
        if not self.evaluation_dashboard:
            self._create_evaluation_dashboard()
        if self.evaluation_dashboard:
            self.evaluation_dashboard.show()
            self.evaluation_dashboard.raise_()
            self.evaluation_dashboard.activateWindow()
            
    def cleanup_evaluation(self) -> None:
        """Cleanup evaluation resources."""
        if self.evaluation_dashboard:
            self.evaluation_dashboard.close()
        self.logger.info("Evaluation resources cleaned up")
        
    def _verify_widget_display(self) -> None:
        """Verify that widgets are properly displayed and not showing screenshots."""
        self.logger.info("=== WIDGET DISPLAY VERIFICATION ===")
        self.logger.info(f"Main window visible: {self.isVisible()}")
        self.logger.info(f"Main window size: {self.size().width()}x{self.size().height()}")
        self.logger.info(f"Central widget visible: {self.central_widget.isVisible()}")
        self.logger.info(f"Main splitter visible: {self.main_splitter.isVisible()}")
        self.logger.info(f"Conversation sidebar visible: {self.conversation_sidebar.isVisible()}")
        self.logger.info(f"Chat area visible: {self.chat_area.isVisible()}")
        self.logger.info(f"Assistant panel visible: {self.assistant_panel.isVisible()}")
        
        # Check if any widgets have screenshot-like content
        self.logger.info("=== CHECKING FOR SCREENSHOT CONTENT ===")
        if hasattr(self.chat_area, 'children'):
            child_count = len(self.chat_area.children())
            self.logger.info(f"Chat area has {child_count} child widgets")
        
        # Force a repaint to ensure proper rendering
        self.repaint()
        self.update()
        self.logger.info("Forced window repaint and update")
        
    def _open_settings(self) -> None:
        """Open the settings window."""
        try:
            if not self.settings_window:
                self.settings_window = SettingsWindow(self)
                self.settings_window.settings_changed.connect(self._handle_settings_changed)
                
            self.settings_window.show()
            self.settings_window.raise_()
            self.settings_window.activateWindow()
            
            self.logger.info("Settings window opened")
            
        except Exception as e:
            self.logger.error(f"Failed to open settings window: {e}")
            
    def _handle_settings_changed(self, settings: Dict[str, Any]) -> None:
        """Handle settings changes from the settings window."""
        try:
            self.logger.info("Settings changed, applying updates...")
            
            # Apply voice settings
            voice_settings = settings.get('voice', {})
            if voice_settings and hasattr(self.assistant_panel, 'apply_voice_settings'):
                self.assistant_panel.apply_voice_settings(voice_settings)
                
            # Apply agent settings
            agent_settings = settings.get('agent', {})
            if agent_settings:
                active_agent = agent_settings.get('active_agent')
                if active_agent and hasattr(self.assistant_panel, 'set_active_agent'):
                    self.assistant_panel.set_active_agent(active_agent)
                    
            # Apply app settings
            app_settings = settings.get('app', {})
            if app_settings:
                theme = app_settings.get('theme')
                if theme and hasattr(self.app, 'theme_manager'):
                    # Apply theme changes
                    pass
                    
            self.update_status("Settings applied successfully")
            self.logger.info("Settings changes applied to application")
            
        except Exception as e:
            self.logger.error(f"Failed to apply settings: {e}")
            self.update_status("Failed to apply settings")
            
    def _show_about(self) -> None:
        """Show about dialog."""
        from PySide6.QtWidgets import QMessageBox
        
        about_text = """
        <h2>Kate LLM Desktop Client</h2>
        <p>Version 1.0.0</p>
        <p>A modern desktop client for multiple LLM providers.</p>
        <p>Built with Python and PySide6.</p>
        """
        
        QMessageBox.about(self, "About Kate", about_text)