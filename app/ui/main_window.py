"""
Main window for Kate LLM Client with 3-column layout.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QFrame, QLabel, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QIcon, QFont
from loguru import logger
from typing import TYPE_CHECKING, Any

from ..core.events import EventBus
from ..core.config import AppSettings
from .components.conversation_sidebar import ConversationSidebar
from .components.chat_area import ChatArea
from .components.assistant_panel import AssistantPanel
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
        
        # Initialize UI components
        self._setup_ui()
        self._setup_layout()
        self._connect_signals()
        
        # Apply theme
        self._apply_theme()
        
        self.logger.info("Main window initialized with 3-column layout")
        
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