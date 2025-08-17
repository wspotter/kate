"""
Message bubble component for Kate LLM Client.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, 
    QTextEdit, QSizePolicy, QPushButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap, QIcon
from loguru import logger
from datetime import datetime
from typing import Optional


class MessageBubble(QFrame):
    """
    Widget for displaying individual chat messages with styling based on role.
    """
    
    # Signals
    copy_requested = Signal(str)  # message content
    
    def __init__(self, role: str, content: str, timestamp: Optional[str] = None):
        super().__init__()
        self.role = role  # "user" or "assistant"
        self.content = content
        self.timestamp = timestamp or datetime.now().strftime("%H:%M")
        
        self._setup_ui()
        self._apply_styling()
        
    def _setup_ui(self) -> None:
        """Set up the message bubble UI."""
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 8, 0, 8)
        
        # Create bubble based on role
        if self.role == "user":
            self._create_user_bubble(main_layout)
        else:
            self._create_assistant_bubble(main_layout)
            
    def _create_user_bubble(self, main_layout: QHBoxLayout) -> None:
        """Create a user message bubble (right-aligned)."""
        # Add stretch to push bubble to the right
        main_layout.addStretch(1)
        
        # Create bubble container
        bubble_container = QFrame()
        bubble_container.setMaximumWidth(500)
        bubble_layout = QVBoxLayout(bubble_container)
        bubble_layout.setContentsMargins(16, 12, 16, 12)
        bubble_layout.setSpacing(4)
        
        # Message content
        content_label = QLabel(self.content)
        content_label.setWordWrap(True)
        content_label.setFont(QFont("Arial", 11))
        content_label.setStyleSheet("color: #ffffff;")
        
        # Timestamp
        time_label = QLabel(self.timestamp)
        time_label.setFont(QFont("Arial", 9))
        time_label.setStyleSheet("color: #cccccc;")
        time_label.setAlignment(Qt.AlignRight)
        
        bubble_layout.addWidget(content_label)
        bubble_layout.addWidget(time_label)
        
        # Set bubble styling
        bubble_container.setStyleSheet("""
            QFrame {
                background-color: #0078d4;
                border-radius: 16px;
                border-bottom-right-radius: 4px;
            }
        """)
        
        main_layout.addWidget(bubble_container)
        
    def _create_assistant_bubble(self, main_layout: QHBoxLayout) -> None:
        """Create an assistant message bubble (left-aligned)."""
        # Create bubble container
        bubble_container = QFrame()
        bubble_container.setMaximumWidth(600)
        bubble_layout = QVBoxLayout(bubble_container)
        bubble_layout.setContentsMargins(16, 12, 16, 12)
        bubble_layout.setSpacing(4)
        
        # Header with avatar and name
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        
        # Avatar (placeholder)
        avatar_label = QLabel("🤖")
        avatar_label.setFont(QFont("Arial", 16))
        avatar_label.setFixedSize(24, 24)
        avatar_label.setAlignment(Qt.AlignCenter)
        
        # Assistant name
        name_label = QLabel("Assistant")
        name_label.setFont(QFont("Arial", 10, QFont.Bold))
        name_label.setStyleSheet("color: #cccccc;")
        
        # Timestamp
        time_label = QLabel(self.timestamp)
        time_label.setFont(QFont("Arial", 9))
        time_label.setStyleSheet("color: #888888;")
        
        header_layout.addWidget(avatar_label)
        header_layout.addWidget(name_label)
        header_layout.addStretch()
        header_layout.addWidget(time_label)
        
        # Message content
        content_label = QLabel(self.content)
        content_label.setWordWrap(True)
        content_label.setFont(QFont("Arial", 11))
        content_label.setStyleSheet("color: #ffffff; line-height: 1.4;")
        content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(4)
        
        copy_button = QPushButton("📋")
        copy_button.setFixedSize(24, 24)
        copy_button.setToolTip("Copy message")
        copy_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #404040;
                border-radius: 4px;
            }
        """)
        copy_button.clicked.connect(self._copy_message)
        
        actions_layout.addWidget(copy_button)
        actions_layout.addStretch()
        
        bubble_layout.addLayout(header_layout)
        bubble_layout.addWidget(content_label)
        bubble_layout.addLayout(actions_layout)
        
        # Set bubble styling
        bubble_container.setStyleSheet("""
            QFrame {
                background-color: #404040;
                border-radius: 16px;
                border-bottom-left-radius: 4px;
            }
        """)
        
        main_layout.addWidget(bubble_container)
        
        # Add stretch to keep bubble left-aligned
        main_layout.addStretch(1)
        
    def _apply_styling(self) -> None:
        """Apply general styling to the message bubble."""
        self.setStyleSheet("""
            MessageBubble {
                background-color: transparent;
                border: none;
            }
        """)
        
    def _copy_message(self) -> None:
        """Copy message content to clipboard."""
        self.copy_requested.emit(self.content)
        
        # Also copy to system clipboard
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(self.content)
        
    def get_content(self) -> str:
        """Get the message content."""
        return self.content
        
    def get_role(self) -> str:
        """Get the message role."""
        return self.role
        
    def get_timestamp(self) -> str:
        """Get the message timestamp."""
        return self.timestamp
        
    def update_content(self, content: str) -> None:
        """Update the message content (for streaming responses)."""
        self.content = content
        # Find and update the content label
        # This is a simplified approach - in a real implementation,
        # you might want to use a more sophisticated method for streaming updates
        pass


class StreamingMessageBubble(MessageBubble):
    """
    Special message bubble for streaming responses.
    """
    
    def __init__(self, role: str):
        super().__init__(role, "", datetime.now().strftime("%H:%M"))
        self.streaming = True
        
    def append_content(self, chunk: str) -> None:
        """Append content chunk for streaming."""
        self.content += chunk
        # Update the display
        # This would need to find and update the content widget
        pass
        
    def finish_streaming(self) -> None:
        """Mark streaming as complete."""
        self.streaming = False
        
    def is_streaming(self) -> bool:
        """Check if message is still streaming."""
        return self.streaming