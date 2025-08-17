"""
Progress indicators for document indexing and background operations.
"""

from typing import Optional

from loguru import logger
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class DocumentIndexingProgressItem(QWidget):
    """Individual progress item for document indexing."""
    
    # Signals
    cancel_requested = Signal(str)  # document_id
    
    def __init__(self, document_id: str, document_name: str, total_chunks: int = 0):
        super().__init__()
        self.document_id = document_id
        self.document_name = document_name
        self.total_chunks = total_chunks
        self.processed_chunks = 0
        
        self._setup_ui()
        
    def _setup_ui(self) -> None:
        """Set up the progress item UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Header with document name and cancel button
        header_layout = QHBoxLayout()
        
        # Document name label
        self.name_label = QLabel(self.document_name)
        self.name_label.setFont(QFont("", 10, QFont.Bold))
        self.name_label.setWordWrap(True)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMaximumWidth(60)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        
        header_layout.addWidget(self.name_label, 1)
        header_layout.addWidget(self.cancel_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(self.total_chunks if self.total_chunks > 0 else 100)
        self.progress_bar.setValue(0)
        
        # Status label
        self.status_label = QLabel("Preparing...")
        self.status_label.setStyleSheet("color: #666; font-size: 9px;")
        
        layout.addLayout(header_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        
        # Style the widget
        self.setStyleSheet("""
            DocumentIndexingProgressItem {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                margin: 2px;
            }
            DocumentIndexingProgressItem:hover {
                background-color: #e9ecef;
            }
        """)
        
    def update_progress(self, processed_chunks: int, status_text: str = "") -> None:
        """Update the progress of this item."""
        self.processed_chunks = processed_chunks
        self.progress_bar.setValue(processed_chunks)
        
        if status_text:
            self.status_label.setText(status_text)
        else:
            if self.total_chunks > 0:
                percentage = (processed_chunks / self.total_chunks) * 100
                self.status_label.setText(f"{processed_chunks}/{self.total_chunks} chunks ({percentage:.1f}%)")
            else:
                self.status_label.setText(f"{processed_chunks} chunks processed")
                
    def set_completed(self, success: bool = True) -> None:
        """Mark the indexing as completed."""
        if success:
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.status_label.setText("✓ Completed")
            self.status_label.setStyleSheet("color: #28a745; font-size: 9px;")
            self.cancel_button.setText("Remove")
        else:
            self.status_label.setText("✗ Failed")
            self.status_label.setStyleSheet("color: #dc3545; font-size: 9px;")
            self.cancel_button.setText("Retry")
            
    def set_error(self, error_message: str) -> None:
        """Set error state."""
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("color: #dc3545; font-size: 9px;")
        self.cancel_button.setText("Retry")
        
    def _on_cancel_clicked(self) -> None:
        """Handle cancel button click."""
        self.cancel_requested.emit(self.document_id)


class DocumentIndexingProgressPanel(QWidget):
    """Panel showing all document indexing progress."""
    
    # Signals
    cancel_indexing = Signal(str)  # document_id
    clear_completed = Signal()
    
    def __init__(self):
        super().__init__()
        self.logger = logger.bind(component="DocumentIndexingProgressPanel")
        self.progress_items = {}  # document_id -> DocumentIndexingProgressItem
        
        self._setup_ui()
        
    def _setup_ui(self) -> None:
        """Set up the progress panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Document Indexing")
        title_label.setFont(QFont("", 12, QFont.Bold))
        
        self.clear_button = QPushButton("Clear Completed")
        self.clear_button.setMaximumWidth(120)
        self.clear_button.clicked.connect(self.clear_completed.emit)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.clear_button)
        
        # Scroll area for progress items
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Container widget for progress items
        self.container_widget = QWidget()
        self.container_layout = QVBoxLayout(self.container_widget)
        self.container_layout.setContentsMargins(4, 4, 4, 4)
        self.container_layout.setSpacing(2)
        self.container_layout.addStretch()  # Push items to top
        
        self.scroll_area.setWidget(self.container_widget)
        
        # Empty state label
        self.empty_label = QLabel("No indexing operations in progress")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #6c757d; font-style: italic; padding: 20px;")
        
        layout.addLayout(header_layout)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.scroll_area)
        
        # Initially hide scroll area
        self.scroll_area.hide()
        
    def add_document_indexing(self, document_id: str, document_name: str, total_chunks: int = 0) -> None:
        """Add a new document indexing operation."""
        if document_id in self.progress_items:
            self.logger.warning(f"Document {document_id} already being indexed")
            return
            
        # Create progress item
        progress_item = DocumentIndexingProgressItem(document_id, document_name, total_chunks)
        progress_item.cancel_requested.connect(self.cancel_indexing.emit)
        
        # Add to layout
        self.container_layout.insertWidget(self.container_layout.count() - 1, progress_item)
        self.progress_items[document_id] = progress_item
        
        # Show scroll area and hide empty label
        self.empty_label.hide()
        self.scroll_area.show()
        
        self.logger.info(f"Added document indexing progress for: {document_name}")
        
    def update_document_progress(self, document_id: str, processed_chunks: int, status_text: str = "") -> None:
        """Update progress for a specific document."""
        if document_id in self.progress_items:
            self.progress_items[document_id].update_progress(processed_chunks, status_text)
            
    def set_document_completed(self, document_id: str, success: bool = True) -> None:
        """Mark document indexing as completed."""
        if document_id in self.progress_items:
            self.progress_items[document_id].set_completed(success)
            self.logger.info(f"Document indexing completed: {document_id} (success: {success})")
            
    def set_document_error(self, document_id: str, error_message: str) -> None:
        """Set error state for document indexing."""
        if document_id in self.progress_items:
            self.progress_items[document_id].set_error(error_message)
            self.logger.error(f"Document indexing error: {document_id} - {error_message}")
            
    def remove_document_progress(self, document_id: str) -> None:
        """Remove a document progress item."""
        if document_id in self.progress_items:
            progress_item = self.progress_items[document_id]
            self.container_layout.removeWidget(progress_item)
            progress_item.deleteLater()
            del self.progress_items[document_id]
            
            # Show empty label if no items left
            if not self.progress_items:
                self.scroll_area.hide()
                self.empty_label.show()
                
            self.logger.info(f"Removed document indexing progress: {document_id}")
            
    def clear_completed_items(self) -> None:
        """Remove all completed progress items."""
        completed_items = []
        for document_id, item in self.progress_items.items():
            if item.status_label.text().startswith("✓"):
                completed_items.append(document_id)
                
        for document_id in completed_items:
            self.remove_document_progress(document_id)
            
        self.logger.info(f"Cleared {len(completed_items)} completed indexing items")
        
    def get_active_indexing_count(self) -> int:
        """Get count of currently active indexing operations."""
        active_count = 0
        for item in self.progress_items.values():
            if not item.status_label.text().startswith(("✓", "✗")):
                active_count += 1
        return active_count


class BackgroundTaskProgressBar(QProgressBar):
    """Global progress bar for background tasks."""
    
    def __init__(self):
        super().__init__()
        self.setVisible(False)
        self.setMaximumHeight(3)
        self.setTextVisible(False)
        
        # Style for thin progress bar
        self.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #e9ecef;
                border-radius: 1px;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 1px;
            }
        """)
        
        # Auto-hide timer
        self.hide_timer = QTimer()
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide)
        
    def show_progress(self, value: int = 0, maximum: int = 100) -> None:
        """Show progress bar with specified values."""
        self.setMaximum(maximum)
        self.setValue(value)
        self.setVisible(True)
        
        # Reset hide timer
        self.hide_timer.stop()
        
    def update_progress(self, value: int) -> None:
        """Update progress value."""
        self.setValue(value)
        if value >= self.maximum():
            # Auto-hide after completion
            self.hide_timer.start(1000)  # Hide after 1 second
            
    def hide_progress(self) -> None:
        """Hide the progress bar."""
        self.setVisible(False)
        self.hide_timer.stop()