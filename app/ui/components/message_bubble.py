"""
Message bubble component for Kate LLM Client.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QPainter,
    QPixmap,
    QSyntaxHighlighter,
    QTextCharFormat,
)

# Optional multimedia imports - fallback gracefully if not available
try:
    from PySide6.QtMultimedia import QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    MULTIMEDIA_AVAILABLE = True
except ImportError:
    QMediaPlayer = None
    QVideoWidget = None
    MULTIMEDIA_AVAILABLE = False

try:
    from PySide6.QtSvgWidgets import QSvgWidget
    SVG_AVAILABLE = True
except ImportError:
    QSvgWidget = None
    SVG_AVAILABLE = False
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class EvaluationIndicator(QWidget):
    """Widget for displaying evaluation scores for assistant messages."""
    
    def __init__(self):
        super().__init__()
        self.evaluation_data: Optional[Dict[str, Any]] = None
        self._setup_ui()
        
    def _setup_ui(self) -> None:
        """Set up the evaluation indicator UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)
        
        # Overall score bar
        score_layout = QHBoxLayout()
        
        self.score_label = QLabel("Quality: --")
        self.score_label.setFont(QFont("Arial", 8))
        self.score_label.setStyleSheet("color: #cccccc;")
        
        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(0)
        self.score_bar.setFixedHeight(4)
        self.score_bar.setVisible(False)
        
        score_layout.addWidget(self.score_label)
        score_layout.addWidget(self.score_bar, 1)
        
        layout.addLayout(score_layout)
        
        # Detailed metrics (initially hidden)
        self.details_label = QLabel("")
        self.details_label.setFont(QFont("Arial", 7))
        self.details_label.setStyleSheet("color: #888888;")
        self.details_label.setVisible(False)
        
        layout.addWidget(self.details_label)
        
    def update_evaluation(self, evaluation_data: Dict[str, Any]) -> None:
        """Update the evaluation display."""
        self.evaluation_data = evaluation_data
        
        # Extract overall score
        overall_score = evaluation_data.get('overall_score', 0.0)
        
        # Update score display
        self.score_label.setText(f"Quality: {overall_score:.2f}")
        self.score_bar.setValue(int(overall_score * 100))
        self.score_bar.setVisible(True)
        
        # Color code the progress bar
        if overall_score >= 0.8:
            color = "#00ff00"  # Green
        elif overall_score >= 0.6:
            color = "#ffaa00"  # Orange
        else:
            color = "#ff4444"  # Red
            
        self.score_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #555555;
                border-radius: 2px;
                background-color: #2b2b2b;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 1px;
            }}
        """)
        
        # Update detailed metrics
        relevance = evaluation_data.get('relevance_score', 0.0)
        coherence = evaluation_data.get('coherence_score', 0.0)
        completeness = evaluation_data.get('completeness_score', 0.0)
        
        details_text = f"R:{relevance:.2f} C:{coherence:.2f} Comp:{completeness:.2f}"
        self.details_label.setText(details_text)
        self.details_label.setVisible(True)
        
    def clear_evaluation(self) -> None:
        """Clear the evaluation display."""
        self.evaluation_data = None
        self.score_label.setText("Quality: --")
        self.score_bar.setVisible(False)
        self.details_label.setVisible(False)
        
    def toggle_details(self) -> None:
        """Toggle the visibility of detailed metrics."""
        if self.evaluation_data:
            self.details_label.setVisible(not self.details_label.isVisible())


class MessageBubble(QFrame):
    """
    Widget for displaying individual chat messages with styling based on role.
    """
    
    # Signals
    copy_requested = Signal(str)  # message content
    evaluation_details_requested = Signal(object)  # evaluation data
    
    def __init__(self, role: str, content: str, timestamp: Optional[str] = None, evaluation_data: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.role = role  # "user" or "assistant"
        self.content = content
        self.timestamp = timestamp or datetime.now().strftime("%H:%M")
        self.evaluation_data = evaluation_data
        self.evaluation_indicator = None
        
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
        
        # Evaluation indicator (for assistant messages)
        if self.role == "assistant":
            self.evaluation_indicator = EvaluationIndicator()
            if self.evaluation_data:
                self.evaluation_indicator.update_evaluation(self.evaluation_data)
            bubble_layout.addWidget(self.evaluation_indicator)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(4)
        
        copy_button = QPushButton("📋")
        copy_button.setFixedSize(24, 24)
        copy_button.setToolTip("Copy message")
        copy_button.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                border: none;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #555555;
                border-radius: 4px;
            }
        """)
        copy_button.clicked.connect(self._copy_message)
        
        # Evaluation details button (for assistant messages with evaluation)
        if self.role == "assistant" and self.evaluation_data:
            eval_button = QPushButton("📊")
            eval_button.setFixedSize(24, 24)
            eval_button.setToolTip("View evaluation details")
            eval_button.setStyleSheet("""
                QPushButton {
                    background-color: #404040;
                    border: none;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #555555;
                    border-radius: 4px;
                }
            """)
            eval_button.clicked.connect(self._show_evaluation_details)
            actions_layout.addWidget(eval_button)
        
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
                background-color: #1e1e1e;
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
        
    def _show_evaluation_details(self) -> None:
        """Show detailed evaluation information."""
        if self.evaluation_data:
            self.evaluation_details_requested.emit(self.evaluation_data)
        
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
        self._update_content_display()
        
    def _update_content_display(self) -> None:
        """Update the content display in the UI."""
        # Find the content label and update it
        if hasattr(self, 'content_label'):
            self.content_label.setText(self.content)
            
    def update_evaluation(self, evaluation_data: Dict[str, Any]) -> None:
        """Update the evaluation data for this message."""
        self.evaluation_data = evaluation_data
        if self.evaluation_indicator:
            self.evaluation_indicator.update_evaluation(evaluation_data)
            
    def get_evaluation_data(self) -> Optional[Dict[str, Any]]:
        """Get the evaluation data for this message."""
        return self.evaluation_data

    def add_rich_content(self, widget: QWidget) -> None:
        """Add a rich content widget to the message bubble."""
        # Find the bubble's main layout and add the widget
        bubble_container = self.findChild(QFrame)
        if bubble_container:
            bubble_layout = bubble_container.layout()
            if bubble_layout:
                # Insert before the actions layout if it exists
                actions_layout = self.findChild(QHBoxLayout)
                if actions_layout:
                    bubble_layout.insertWidget(bubble_layout.indexOf(actions_layout), widget)
                else:
                    bubble_layout.addWidget(widget)


class StreamingMessageBubble(MessageBubble):
    """
    Special message bubble for streaming responses with real-time updates.
    """
    
    def __init__(self, role: str):
        super().__init__(role, "▋", datetime.now().strftime("%H:%M"))  # Start with cursor
        self.streaming = True
        self.content_label = None  # Will be set during UI setup
        self.pending_evaluation = None  # Store evaluation to apply after streaming
        
    def _create_assistant_bubble(self, main_layout: QHBoxLayout) -> None:
        """Create an assistant message bubble with streaming support."""
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
        
        # Streaming indicator
        self.streaming_indicator = QLabel("⚡ Streaming...")
        self.streaming_indicator.setFont(QFont("Arial", 9))
        self.streaming_indicator.setStyleSheet("color: #ffaa00;")
        
        # Timestamp
        time_label = QLabel(self.timestamp)
        time_label.setFont(QFont("Arial", 9))
        time_label.setStyleSheet("color: #888888;")
        
        header_layout.addWidget(avatar_label)
        header_layout.addWidget(name_label)
        header_layout.addWidget(self.streaming_indicator)
        header_layout.addStretch()
        header_layout.addWidget(time_label)
        
        # Message content (use QLabel for real-time updates)
        self.content_label = QLabel(self.content)
        self.content_label.setWordWrap(True)
        self.content_label.setFont(QFont("Arial", 11))
        self.content_label.setStyleSheet("color: #ffffff; line-height: 1.4;")
        self.content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # Evaluation indicator (initially hidden during streaming)
        self.evaluation_indicator = EvaluationIndicator()
        self.evaluation_indicator.setVisible(False)
        
        # Action buttons (initially hidden during streaming)
        self.actions_layout = QHBoxLayout()
        self.actions_layout.setSpacing(4)
        
        self.copy_button = QPushButton("📋")
        self.copy_button.setFixedSize(24, 24)
        self.copy_button.setToolTip("Copy message")
        self.copy_button.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                border: none;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #555555;
                border-radius: 4px;
            }
        """)
        self.copy_button.clicked.connect(self._copy_message)
        self.copy_button.setVisible(False)  # Hidden during streaming
        
        # Evaluation details button (hidden during streaming)
        self.eval_button = QPushButton("📊")
        self.eval_button.setFixedSize(24, 24)
        self.eval_button.setToolTip("View evaluation details")
        self.eval_button.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                border: none;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #555555;
                border-radius: 4px;
            }
        """)
        self.eval_button.clicked.connect(self._show_evaluation_details)
        self.eval_button.setVisible(False)  # Hidden during streaming
        
        self.actions_layout.addWidget(self.eval_button)
        self.actions_layout.addWidget(self.copy_button)
        self.actions_layout.addStretch()
        
        bubble_layout.addLayout(header_layout)
        bubble_layout.addWidget(self.content_label)
        bubble_layout.addWidget(self.evaluation_indicator)
        bubble_layout.addLayout(self.actions_layout)
        
        # Set bubble styling with streaming animation
        bubble_container.setStyleSheet("""
            QFrame {
                background-color: #404040;
                border-radius: 16px;
                border-bottom-left-radius: 4px;
                border-left: 3px solid #ffaa00;
            }
        """)
        
        main_layout.addWidget(bubble_container)
        main_layout.addStretch(1)
        
    def append_content(self, chunk: str) -> None:
        """Append content chunk for streaming."""
        if self.streaming:
            # Remove cursor if it exists
            if self.content.endswith("▋"):
                self.content = self.content[:-1]
            
            # Append new chunk
            self.content += chunk
            
            # Add cursor back
            display_content = self.content + "▋"
            
            # Update the display
            if self.content_label:
                self.content_label.setText(display_content)
                
    def finish_streaming(self) -> None:
        """Mark streaming as complete."""
        self.streaming = False
        
        # Remove cursor
        if self.content.endswith("▋"):
            self.content = self.content[:-1]
            
        # Update final content
        if self.content_label:
            self.content_label.setText(self.content)
        else:
            logger.warning("StreamingMessageBubble.finish_streaming called before content_label initialized")
            
        # Hide streaming indicator and show actions
        if hasattr(self, 'streaming_indicator'):
            self.streaming_indicator.setVisible(False)
        if hasattr(self, 'copy_button'):
            self.copy_button.setVisible(True)
            
        # Show evaluation indicator if we have pending evaluation data
        if self.pending_evaluation:
            self.evaluation_data = self.pending_evaluation
            self.evaluation_indicator.update_evaluation(self.pending_evaluation)
            self.evaluation_indicator.setVisible(True)
            self.eval_button.setVisible(True)
            self.pending_evaluation = None
            
        # Update styling to remove streaming border
        if self.content_label:
            parent_frame = self.content_label.parent()
            if parent_frame:
                self.evaluation_indicator.setVisible(True)
                parent_frame.setStyleSheet("""
                    QFrame {
                        background-color: #404040;
                        border-radius: 16px;
                        border-bottom-left-radius: 4px;
                    }
                """)
            
    def set_pending_evaluation(self, evaluation_data: Dict[str, Any]) -> None:
        """Set evaluation data to be shown after streaming completes."""
        if self.streaming:
            self.pending_evaluation = evaluation_data
        else:
            self.update_evaluation(evaluation_data)
        
    def is_streaming(self) -> bool:
        """Check if message is still streaming."""
        return self.streaming


class ImageContentWidget(QWidget):
    """Widget for displaying image content."""
    def __init__(self, file_path: str):
        super().__init__()
        # In a real implementation, load and display the image
        layout = QVBoxLayout(self)
        label = QLabel(f"[Image: {file_path}]")
        label.setStyleSheet("color: #aaa;")
        layout.addWidget(label)


class AudioContentWidget(QWidget):
    """Widget for displaying audio content."""
    def __init__(self, file_path: str):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel(f"[Audio: {file_path}]")
        label.setStyleSheet("color: #aaa;")
        play_button = QPushButton("Play")
        layout.addWidget(label)
        layout.addWidget(play_button)


class CodeContentWidget(QWidget):
    """Widget for displaying code content with syntax highlighting."""
    def __init__(self, language: str, code: str):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel(f"[{language.capitalize()} Code Snippet]")
        label.setStyleSheet("font-weight: bold; color: #ccc;")
        
        code_view = QTextEdit()
        code_view.setPlainText(code)
        code_view.setReadOnly(True)
        # In a real implementation, a QSyntaxHighlighter would be used here
        
        layout.addWidget(label)
        layout.addWidget(code_view)