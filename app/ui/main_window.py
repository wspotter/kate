"""
Main window for Kate LLM Client with 3-column layout.
"""

from typing import TYPE_CHECKING, Any

from loguru import logger
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core.config import AppSettings
from ..core.events import EventBus
from ..services.embedding_service import EmbeddingService
from ..services.rag_evaluation_service import RAGEvaluationService
from .components.assistant_panel import AssistantPanel
from .components.chat_area import ChatArea
from .components.conversation_sidebar import ConversationSidebar
from .components.evaluation_dashboard import EvaluationDashboard
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
    
    def __init__(self, app: KateApplication, settings: AppSettings, event_bus: EventBus):
        super().__init__()
        self.app = app
        self.settings = settings
        self.event_bus = event_bus
        self.logger = logger.bind(component="MainWindow")
        
        # Initialize evaluation service
        self.evaluation_service = None
        self.evaluation_dashboard = None
        
        # Initialize UI components
        self._setup_ui()
        self._setup_layout()
        self._connect_signals()
        self._setup_evaluation_service()
        
        # Apply theme
        self._apply_theme()
        
        self.logger.info("Main window initialized with 3-column layout and evaluation integration")
        
    def _setup_ui(self) -> None:
        """Set up the main window UI."""
        # Window properties
        self.setWindowTitle("Kate - LLM Desktop Client")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Set window icon
        # self.setWindowIcon(QIcon(":/icons/kate.png"))  # TODO: Add icon
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
    def _setup_layout(self) -> None:
        """Set up the 3-column layout."""
        # Create main splitter for 3 columns
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        
        # Left panel - Conversation sidebar
        self.conversation_sidebar = ConversationSidebar(self.event_bus)
        self.conversation_sidebar.setMinimumWidth(250)
        self.conversation_sidebar.setMaximumWidth(400)
        
        # Center panel - Chat area
        self.chat_area = ChatArea(self.event_bus)
        
        # Right panel - Assistant panel
        self.assistant_panel = AssistantPanel(self.event_bus)
        self.assistant_panel.setMinimumWidth(280)
        self.assistant_panel.setMaximumWidth(350)
        
        # Add panels to splitter
        self.main_splitter.addWidget(self.conversation_sidebar)
        self.main_splitter.addWidget(self.chat_area)
        self.main_splitter.addWidget(self.assistant_panel)
        
        # Set splitter proportions (25% | 50% | 25%)
        self.main_splitter.setSizes([300, 700, 300])
        
        # Add splitter to main layout
        self.main_layout.addWidget(self.main_splitter)
        
        # Create status bar
        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)
        
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
        # Assistant change handling will be implemented
        
    def _handle_evaluation_received(self, evaluation_data) -> None:
        """Handle evaluation data received from chat area."""
        try:
            # Update assistant panel with evaluation metrics
            if hasattr(evaluation_data, 'to_dict'):
                # ResponseEvaluation object
                eval_dict = evaluation_data.to_dict()
                self.assistant_panel.update_evaluation(evaluation_data)
            else:
                # Already a dictionary
                eval_dict = evaluation_data
                
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
            self.evaluation_dashboard.show()
            self.evaluation_dashboard.raise_()
            self.evaluation_dashboard.activateWindow()
            
            self.logger.info("Evaluation dashboard opened")
            
        except Exception as e:
            self.logger.error(f"Failed to show evaluation dashboard: {e}")
            
    def _setup_evaluation_service(self) -> None:
        """Set up the evaluation service and connect it to components."""
        try:
            # Create a minimal embedding service for evaluation
            # In a full implementation, this would be injected from the application
            embedding_service = EmbeddingService()
            
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
        
    def get_evaluation_service(self) -> RAGEvaluationService:
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