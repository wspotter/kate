"""
Status bar component for Kate LLM Client.
"""

from PySide6.QtWidgets import QStatusBar, QLabel, QProgressBar, QHBoxLayout, QWidget
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from loguru import logger


class StatusBar(QStatusBar):
    """
    Custom status bar for the Kate application.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logger.bind(component="StatusBar")
        
        self._setup_ui()
        self._setup_styling()
        
    def _setup_ui(self) -> None:
        """Set up the status bar UI."""
        # Main status message
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Arial", 10))
        self.addWidget(self.status_label, 1)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.addWidget(self.progress_bar)
        
        # Connection status
        self.connection_label = QLabel("Disconnected")
        self.connection_label.setFont(QFont("Arial", 10))
        self.connection_label.setMinimumWidth(100)
        self.addPermanentWidget(self.connection_label)
        
        # Provider info
        self.provider_label = QLabel("No Provider")
        self.provider_label.setFont(QFont("Arial", 10))
        self.provider_label.setMinimumWidth(120)
        self.addPermanentWidget(self.provider_label)
        
    def _setup_styling(self) -> None:
        """Apply styling to the status bar."""
        self.setStyleSheet("""
            QStatusBar {
                background-color: #2b2b2b;
                border-top: 1px solid #404040;
                color: #ffffff;
                padding: 2px;
            }
            
            QLabel {
                color: #ffffff;
                padding: 2px 8px;
            }
            
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #404040;
                text-align: center;
                color: #ffffff;
            }
            
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
        """)
        
    def set_status(self, message: str, timeout: int = 0) -> None:
        """Set the main status message."""
        self.status_label.setText(message)
        if timeout > 0:
            QTimer.singleShot(timeout, lambda: self.status_label.setText("Ready"))
            
    def set_connection_status(self, connected: bool) -> None:
        """Set the connection status."""
        if connected:
            self.connection_label.setText("Connected")
            self.connection_label.setStyleSheet("QLabel { color: #00ff00; }")
        else:
            self.connection_label.setText("Disconnected")
            self.connection_label.setStyleSheet("QLabel { color: #ff6666; }")
            
    def set_provider_info(self, provider: str, model: str = "") -> None:
        """Set the current provider information."""
        if model:
            text = f"{provider} ({model})"
        else:
            text = provider
        self.provider_label.setText(text)
        
    def show_progress(self, show: bool = True) -> None:
        """Show or hide the progress bar."""
        if show:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()
            
    def set_progress(self, value: int) -> None:
        """Set the progress bar value (0-100)."""
        self.progress_bar.setValue(value)
        
    def set_progress_text(self, text: str) -> None:
        """Set the progress bar text."""
        self.progress_bar.setFormat(text)
        
    def show_loading(self, message: str = "Loading...") -> None:
        """Show loading indicator."""
        self.set_status(message)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.show_progress(True)
        
    def hide_loading(self) -> None:
        """Hide loading indicator."""
        self.progress_bar.setRange(0, 100)  # Reset to determinate
        self.show_progress(False)
        self.set_status("Ready")