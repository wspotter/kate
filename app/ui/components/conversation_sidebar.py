"""
Conversation sidebar component for Kate LLM Client.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLineEdit, QLabel, QFrame, QMenu, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QIcon, QFont, QAction
from loguru import logger
from typing import List, Optional, Dict, Any

from ...core.events import EventBus


class ConversationItem(QListWidgetItem):
    """Custom list item for conversations."""
    
    def __init__(self, conversation_id: str, title: str, last_message: str = ""):
        super().__init__()
        self.conversation_id = conversation_id
        self.title = title
        self.last_message = last_message
        self._update_display()
        
    def _update_display(self) -> None:
        """Update the display text for this item."""
        display_text = f"{self.title}\n{self.last_message[:50]}..." if self.last_message else self.title
        self.setText(display_text)
        
    def update_title(self, title: str) -> None:
        """Update the conversation title."""
        self.title = title
        self._update_display()
        
    def update_last_message(self, message: str) -> None:
        """Update the last message preview."""
        self.last_message = message
        self._update_display()


class ConversationSidebar(QWidget):
    """
    Sidebar widget for managing conversations.
    """
    
    # Signals
    conversation_selected = Signal(str)  # conversation_id
    conversation_created = Signal()
    conversation_deleted = Signal(str)  # conversation_id
    
    def __init__(self, event_bus: EventBus):
        super().__init__()
        self.event_bus = event_bus
        self.logger = logger.bind(component="ConversationSidebar")
        
        self.conversations: Dict[str, ConversationItem] = {}
        self.current_conversation_id: Optional[str] = None
        
        self._setup_ui()
        self._connect_signals()
        self._load_conversations()
        
    def _setup_ui(self) -> None:
        """Set up the sidebar UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Header
        header_layout = QHBoxLayout()
        
        self.title_label = QLabel("Conversations")
        self.title_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        self.new_button = QPushButton("+")
        self.new_button.setFixedSize(30, 30)
        self.new_button.setToolTip("New Conversation")
        
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.new_button)
        
        layout.addLayout(header_layout)
        
        # Search bar
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search conversations...")
        layout.addWidget(self.search_edit)
        
        # Conversation list
        self.conversation_list = QListWidget()
        self.conversation_list.setContextMenuPolicy(Qt.CustomContextMenu)
        layout.addWidget(self.conversation_list)
        
        # Apply styling
        self._apply_styling()
        
    def _apply_styling(self) -> None:
        """Apply custom styling to the sidebar."""
        self.setStyleSheet("""
            ConversationSidebar {
                background-color: #2b2b2b;
                border-right: 1px solid #404040;
            }
            
            QLabel {
                color: #ffffff;
            }
            
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 15px;
                color: #ffffff;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #505050;
            }
            
            QPushButton:pressed {
                background-color: #353535;
            }
            
            QLineEdit {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px;
                color: #ffffff;
            }
            
            QLineEdit:focus {
                border-color: #0078d4;
            }
            
            QListWidget {
                background-color: #2b2b2b;
                border: none;
                outline: none;
            }
            
            QListWidget::item {
                background-color: #2b2b2b;
                border: none;
                padding: 12px 8px;
                border-bottom: 1px solid #404040;
                color: #ffffff;
            }
            
            QListWidget::item:hover {
                background-color: #404040;
            }
            
            QListWidget::item:selected {
                background-color: #0078d4;
            }
        """)
        
    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.new_button.clicked.connect(self.create_new_conversation)
        self.conversation_list.itemClicked.connect(self._on_conversation_clicked)
        self.conversation_list.customContextMenuRequested.connect(self._show_context_menu)
        self.search_edit.textChanged.connect(self._filter_conversations)
        
    def _load_conversations(self) -> None:
        """Load conversations from database."""
        # TODO: Load real conversations from database
        # For now, add some sample conversations
        self._add_sample_conversations()
        
    def _add_sample_conversations(self) -> None:
        """Add sample conversations for testing."""
        sample_conversations = [
            {
                "id": "conv_1",
                "title": "Python Development Help",
                "last_message": "How do I create a virtual environment?"
            },
            {
                "id": "conv_2", 
                "title": "Machine Learning Questions",
                "last_message": "Explain neural networks"
            },
            {
                "id": "conv_3",
                "title": "Web Development",
                "last_message": "Best practices for React"
            }
        ]
        
        for conv in sample_conversations:
            self.add_conversation(conv["id"], conv["title"], conv["last_message"])
            
    def add_conversation(self, conversation_id: str, title: str, last_message: str = "") -> None:
        """Add a new conversation to the sidebar."""
        item = ConversationItem(conversation_id, title, last_message)
        self.conversations[conversation_id] = item
        self.conversation_list.addItem(item)
        self.logger.debug(f"Added conversation: {title}")
        
    def remove_conversation(self, conversation_id: str) -> None:
        """Remove a conversation from the sidebar."""
        if conversation_id in self.conversations:
            item = self.conversations[conversation_id]
            row = self.conversation_list.row(item)
            self.conversation_list.takeItem(row)
            del self.conversations[conversation_id]
            self.logger.debug(f"Removed conversation: {conversation_id}")
            
    def select_conversation(self, conversation_id: str) -> None:
        """Select a specific conversation."""
        if conversation_id in self.conversations:
            item = self.conversations[conversation_id]
            self.conversation_list.setCurrentItem(item)
            self.current_conversation_id = conversation_id
            
    def create_new_conversation(self) -> None:
        """Create a new conversation."""
        self.logger.debug("Creating new conversation")
        
        # Generate a new conversation ID
        conversation_id = f"conv_{len(self.conversations) + 1}"
        title = f"New Chat {len(self.conversations) + 1}"
        
        # Add to sidebar
        self.add_conversation(conversation_id, title)
        
        # Select the new conversation
        self.select_conversation(conversation_id)
        
        # Emit signals
        self.conversation_created.emit()
        self.conversation_selected.emit(conversation_id)
        
    def _on_conversation_clicked(self, item: ConversationItem) -> None:
        """Handle conversation item click."""
        if isinstance(item, ConversationItem):
            self.current_conversation_id = item.conversation_id
            self.conversation_selected.emit(item.conversation_id)
            self.logger.debug(f"Selected conversation: {item.conversation_id}")
            
    def _show_context_menu(self, position) -> None:
        """Show context menu for conversation items."""
        item = self.conversation_list.itemAt(position)
        if not isinstance(item, ConversationItem):
            return
            
        menu = QMenu(self)
        
        # Rename action
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self._rename_conversation(item))
        menu.addAction(rename_action)
        
        # Delete action
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._delete_conversation(item))
        menu.addAction(delete_action)
        
        menu.exec(self.conversation_list.mapToGlobal(position))
        
    def _rename_conversation(self, item: ConversationItem) -> None:
        """Rename a conversation."""
        # TODO: Implement rename dialog
        self.logger.debug(f"Rename conversation: {item.conversation_id}")
        
    def _delete_conversation(self, item: ConversationItem) -> None:
        """Delete a conversation with confirmation."""
        reply = QMessageBox.question(
            self, 
            "Delete Conversation",
            f"Are you sure you want to delete '{item.title}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.remove_conversation(item.conversation_id)
            self.conversation_deleted.emit(item.conversation_id)
            
    def _filter_conversations(self, text: str) -> None:
        """Filter conversations based on search text."""
        for i in range(self.conversation_list.count()):
            item = self.conversation_list.item(i)
            if isinstance(item, ConversationItem):
                # Show/hide based on title match
                visible = text.lower() in item.title.lower()
                item.setHidden(not visible)
                
    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        """Update a conversation's title."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].update_title(title)
            
    def update_conversation_preview(self, conversation_id: str, message: str) -> None:
        """Update a conversation's last message preview."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].update_last_message(message)