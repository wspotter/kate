"""
Chat area component for Kate LLM Client.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QScrollArea, 
    QFrame, QPushButton, QLabel, QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread
from PySide6.QtGui import QFont, QTextCursor, QKeySequence
from loguru import logger
from typing import List, Optional, Dict, Any

from ...core.events import EventBus
from .message_bubble import MessageBubble


class ChatScrollArea(QScrollArea):
    """Custom scroll area for chat messages."""
    
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(16, 16, 16, 16)
        self.content_layout.setSpacing(12)
        self.content_layout.addStretch()  # Push messages to bottom initially
        
        self.setWidget(self.content_widget)
        
    def add_message_widget(self, message_widget: MessageBubble) -> None:
        """Add a message widget to the chat area."""
        # Insert before the stretch
        count = self.content_layout.count()
        self.content_layout.insertWidget(count - 1, message_widget)
        
        # Auto-scroll to bottom
        QTimer.singleShot(10, self._scroll_to_bottom)
        
    def _scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the chat area."""
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def clear_messages(self) -> None:
        """Clear all messages from the chat area."""
        # Remove all widgets except the stretch
        while self.content_layout.count() > 1:
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


class MessageInputArea(QWidget):
    """Message input area with send button."""
    
    # Signals
    message_sent = Signal(str)
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self) -> None:
        """Set up the input area UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 16)
        layout.setSpacing(8)
        
        # Message input
        self.message_input = QTextEdit()
        self.message_input.setPlaceholderText("Type your message here... (Ctrl+Enter to send)")
        self.message_input.setMaximumHeight(120)
        self.message_input.setMinimumHeight(40)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.setFixedSize(80, 40)
        self.send_button.setEnabled(False)
        
        layout.addWidget(self.message_input, 1)
        layout.addWidget(self.send_button)
        
        # Apply styling
        self._apply_styling()
        
    def _apply_styling(self) -> None:
        """Apply styling to the input area."""
        self.setStyleSheet("""
            MessageInputArea {
                background-color: #2b2b2b;
                border-top: 1px solid #404040;
            }
            
            QTextEdit {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 12px;
                color: #ffffff;
                font-size: 14px;
                line-height: 1.4;
            }
            
            QTextEdit:focus {
                border-color: #0078d4;
            }
            
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 8px;
                color: #ffffff;
                font-weight: bold;
                font-size: 14px;
            }
            
            QPushButton:hover {
                background-color: #106ebe;
            }
            
            QPushButton:pressed {
                background-color: #005a9e;
            }
            
            QPushButton:disabled {
                background-color: #404040;
                color: #888888;
            }
        """)
        
    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.send_button.clicked.connect(self._send_message)
        self.message_input.textChanged.connect(self._on_text_changed)
        
    def _send_message(self) -> None:
        """Send the current message."""
        text = self.message_input.toPlainText().strip()
        if text:
            self.message_sent.emit(text)
            self.message_input.clear()
            
    def _on_text_changed(self) -> None:
        """Handle text input changes."""
        has_text = bool(self.message_input.toPlainText().strip())
        self.send_button.setEnabled(has_text)
        
    def keyPressEvent(self, event) -> None:
        """Handle key press events."""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if event.modifiers() & Qt.ControlModifier:
                # Ctrl+Enter sends message
                self._send_message()
                return
        super().keyPressEvent(event)
        
    def set_focus(self) -> None:
        """Set focus to the message input."""
        self.message_input.setFocus()


class ChatArea(QWidget):
    """
    Main chat area widget for displaying conversations and message input.
    """
    
    # Signals
    message_sent = Signal(str)
    
    def __init__(self, event_bus: EventBus):
        super().__init__()
        self.event_bus = event_bus
        self.logger = logger.bind(component="ChatArea")
        
        self.current_conversation_id: Optional[str] = None
        self.messages: List[MessageBubble] = []
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self) -> None:
        """Set up the chat area UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        self.header = self._create_header()
        layout.addWidget(self.header)
        
        # Chat scroll area
        self.chat_scroll = ChatScrollArea()
        layout.addWidget(self.chat_scroll, 1)
        
        # Message input area
        self.input_area = MessageInputArea()
        layout.addWidget(self.input_area)
        
        # Apply styling
        self._apply_styling()
        
    def _create_header(self) -> QWidget:
        """Create the chat header."""
        header = QFrame()
        header.setFixedHeight(60)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)
        
        # Title and info
        self.title_label = QLabel("Select a conversation")
        self.title_label.setFont(QFont("Arial", 14, QFont.Bold))
        
        self.info_label = QLabel("")
        self.info_label.setFont(QFont("Arial", 10))
        
        title_layout = QVBoxLayout()
        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.info_label)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        
        return header
        
    def _apply_styling(self) -> None:
        """Apply styling to the chat area."""
        self.setStyleSheet("""
            ChatArea {
                background-color: #1e1e1e;
            }
            
            QFrame {
                background-color: #2b2b2b;
                border-bottom: 1px solid #404040;
            }
            
            QLabel {
                color: #ffffff;
            }
        """)
        
    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.input_area.message_sent.connect(self._handle_message_sent)
        
    def _handle_message_sent(self, message: str) -> None:
        """Handle message sent from input area."""
        if not self.current_conversation_id:
            self.logger.warning("No conversation selected")
            return
            
        # Add user message to chat
        self.add_message("user", message)
        
        # Emit signal for external handling
        self.message_sent.emit(message)
        
        # Add a placeholder assistant response
        self._add_placeholder_response()
        
    def _add_placeholder_response(self) -> None:
        """Add a placeholder AI response."""
        # This will be replaced with real LLM integration
        placeholder_responses = [
            "I understand your question. Let me help you with that.",
            "That's an interesting point. Here's my perspective...",
            "I'd be happy to assist you with this task.",
            "Let me think about this and provide you with a detailed response."
        ]
        
        import random
        response = random.choice(placeholder_responses)
        
        # Add after a short delay to simulate processing
        QTimer.singleShot(1000, lambda: self.add_message("assistant", response))
        
    def load_conversation(self, conversation_id: str) -> None:
        """Load a conversation into the chat area."""
        self.logger.debug(f"Loading conversation: {conversation_id}")
        
        self.current_conversation_id = conversation_id
        
        # Update header
        self.title_label.setText(f"Conversation {conversation_id}")
        self.info_label.setText("Ready to chat")
        
        # Clear existing messages
        self.clear_messages()
        
        # Load conversation messages (placeholder for now)
        self._load_sample_messages()
        
        # Focus input area
        self.input_area.set_focus()
        
    def _load_sample_messages(self) -> None:
        """Load sample messages for testing."""
        if self.current_conversation_id == "conv_1":
            self.add_message("user", "How do I create a virtual environment in Python?")
            self.add_message("assistant", "To create a virtual environment in Python, you can use the `venv` module. Here's how:\n\n1. Open your terminal\n2. Navigate to your project directory\n3. Run: `python -m venv myenv`\n4. Activate it with `myenv\\Scripts\\activate` (Windows) or `source myenv/bin/activate` (Mac/Linux)")
        elif self.current_conversation_id == "conv_2":
            self.add_message("user", "Can you explain neural networks?")
            self.add_message("assistant", "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process and transmit information. Each connection has a weight that adjusts during learning to improve the network's ability to recognize patterns and make predictions.")
            
    def add_message(self, role: str, content: str, timestamp: Optional[str] = None) -> None:
        """Add a message to the chat area."""
        message_bubble = MessageBubble(role, content, timestamp)
        self.messages.append(message_bubble)
        self.chat_scroll.add_message_widget(message_bubble)
        
        self.logger.debug(f"Added {role} message: {content[:50]}...")
        
    def clear_messages(self) -> None:
        """Clear all messages from the chat area."""
        self.chat_scroll.clear_messages()
        self.messages.clear()
        
    def get_conversation_id(self) -> Optional[str]:
        """Get the current conversation ID."""
        return self.current_conversation_id
        
    def set_loading(self, loading: bool) -> None:
        """Set loading state for the chat area."""
        self.input_area.setEnabled(not loading)
        if loading:
            self.info_label.setText("AI is thinking...")
        else:
            self.info_label.setText("Ready to chat")