"""
Document management UI components for Kate LLM Client.

Provides comprehensive document upload, organization, and metadata viewing
capabilities for the RAG system.
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.events import EventBus
from ...database.manager import DatabaseManager
from ...database.models import Document, DocumentChunk
from ...services.document_processor import DocumentProcessor

# Lazy import to avoid hanging during startup
# from ...services.embedding_service import EmbeddingService


@dataclass
class DocumentInfo:
    """Document information for UI display."""
    id: str
    title: str
    filename: str
    file_type: str
    file_size: int
    created_at: datetime
    processed: bool
    chunk_count: int
    metadata: Dict[str, Any]


class DocumentUploadThread(QThread):
    """Thread for background document processing."""
    
    progress_updated = Signal(int, str)  # progress, status
    document_processed = Signal(str, bool, str)  # document_id, success, message
    
    def __init__(self, file_paths: List[str], document_processor: DocumentProcessor):
        super().__init__()
        self.file_paths = file_paths
        self.document_processor = document_processor
        self._stop_requested = False
        
    def run(self):
        """Process documents in background thread."""
        total_files = len(self.file_paths)
        
        for i, file_path in enumerate(self.file_paths):
            if self._stop_requested:
                break
                
            try:
                # Update progress
                progress = int((i / total_files) * 100)
                self.progress_updated.emit(progress, f"Processing {Path(file_path).name}")
                
                # Process document (this would be async in real implementation)
                # For now, simulate processing
                import time
                time.sleep(0.5)  # Simulate processing time
                
                # Emit success
                self.document_processed.emit(
                    f"doc_{i}", True, f"Successfully processed {Path(file_path).name}"
                )
                
            except Exception as e:
                # Emit error
                self.document_processed.emit(
                    f"doc_{i}", False, f"Failed to process {Path(file_path).name}: {str(e)}"
                )
                
        # Complete
        self.progress_updated.emit(100, "Processing complete")
        
    def stop_processing(self):
        """Stop the processing thread."""
        self._stop_requested = True


class DocumentUploadWidget(QWidget):
    """Widget for uploading and processing documents."""
    
    # Signals
    documents_uploaded = Signal(list)  # List of document paths
    upload_progress = Signal(int, str)  # progress, status
    
    def __init__(self):
        super().__init__()
        self.upload_thread: Optional[DocumentUploadThread] = None
        self.document_processor: Optional[DocumentProcessor] = None
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        """Set up the upload widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Title
        title_label = QLabel("Document Upload")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # Upload area (drag & drop)
        self.upload_area = QFrame()
        self.upload_area.setFrameStyle(QFrame.StyledPanel)
        self.upload_area.setLineWidth(2)
        self.upload_area.setAcceptDrops(True)
        self.upload_area.setMinimumHeight(120)
        
        upload_layout = QVBoxLayout(self.upload_area)
        upload_layout.setAlignment(Qt.AlignCenter)
        
        # Upload icon and text
        upload_label = QLabel("📄")
        upload_label.setFont(QFont("Arial", 32))
        upload_label.setAlignment(Qt.AlignCenter)
        
        instruction_label = QLabel("Drop files here or click to browse")
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet("color: #888888;")
        
        # Browse button
        self.browse_button = QPushButton("Browse Files")
        self.browse_button.setMaximumWidth(120)
        
        upload_layout.addWidget(upload_label)
        upload_layout.addWidget(instruction_label)
        upload_layout.addWidget(self.browse_button, 0, Qt.AlignCenter)
        
        layout.addWidget(self.upload_area)
        
        # Supported formats
        formats_label = QLabel("Supported: PDF, DOCX, TXT, HTML, CSV, JSON, Excel")
        formats_label.setStyleSheet("color: #666666; font-size: 10px;")
        formats_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(formats_label)
        
        # Progress section
        progress_group = QGroupBox("Upload Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        # Apply styling
        self._apply_styling()
        
    def _apply_styling(self):
        """Apply styling to the upload widget."""
        self.setStyleSheet("""
            DocumentUploadWidget {
                background-color: #2b2b2b;
            }
            
            QFrame {
                background-color: #404040;
                border: 2px dashed #666666;
                border-radius: 8px;
            }
            
            QFrame:hover {
                border-color: #0078d4;
                background-color: #4a4a4a;
            }
            
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #106ebe;
            }
            
            QPushButton:pressed {
                background-color: #005a9e;
            }
            
            QLabel {
                color: #ffffff;
            }
            
            QGroupBox {
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 4px;
                margin: 8px 0px;
                padding-top: 8px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 8px 0 8px;
            }
            
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                background-color: #404040;
                text-align: center;
            }
            
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
        """)
        
    def _connect_signals(self):
        """Connect widget signals."""
        self.browse_button.clicked.connect(self._browse_files)
        self.cancel_button.clicked.connect(self._cancel_upload)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        urls = event.mimeData().urls()
        file_paths = []
        
        for url in urls:
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path):
                    file_paths.append(file_path)
                    
        if file_paths:
            self._process_files(file_paths)
            
    def _browse_files(self):
        """Open file browser dialog."""
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter(
            "Documents (*.pdf *.docx *.txt *.html *.csv *.json *.xlsx *.xls);;All Files (*)"
        )
        
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            self._process_files(file_paths)
            
    def _process_files(self, file_paths: List[str]):
        """Process the selected files."""
        if not file_paths:
            return
            
        # Show progress UI
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.cancel_button.setVisible(True)
        self.browse_button.setEnabled(False)
        
        # Start upload thread
        if self.document_processor:
            self.upload_thread = DocumentUploadThread(file_paths, self.document_processor)
            self.upload_thread.progress_updated.connect(self._on_progress_updated)
            self.upload_thread.document_processed.connect(self._on_document_processed)
            self.upload_thread.finished.connect(self._on_upload_finished)
            self.upload_thread.start()
        else:
            # Fallback without document processor
            self._simulate_upload(file_paths)
            
    def _simulate_upload(self, file_paths: List[str]):
        """Simulate upload for testing."""
        self.status_label.setText(f"Processing {len(file_paths)} files...")
        self.progress_bar.setValue(50)
        
        # Use timer to simulate completion
        QTimer.singleShot(2000, lambda: self._complete_simulation(file_paths))
        
    def _complete_simulation(self, file_paths: List[str]):
        """Complete simulated upload."""
        self.progress_bar.setValue(100)
        self.status_label.setText("Upload complete!")
        self.documents_uploaded.emit(file_paths)
        
        QTimer.singleShot(1000, self._reset_ui)
        
    def _on_progress_updated(self, progress: int, status: str):
        """Handle progress update."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        self.upload_progress.emit(progress, status)
        
    def _on_document_processed(self, document_id: str, success: bool, message: str):
        """Handle document processing completion."""
        if success:
            logger.info(f"Document processed successfully: {message}")
        else:
            logger.error(f"Document processing failed: {message}")
            
    def _on_upload_finished(self):
        """Handle upload thread completion."""
        self._reset_ui()
        
    def _cancel_upload(self):
        """Cancel the current upload."""
        if self.upload_thread and self.upload_thread.isRunning():
            self.upload_thread.stop_processing()
            self.upload_thread.wait()
            
        self._reset_ui()
        
    def _reset_ui(self):
        """Reset the UI to initial state."""
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.cancel_button.setVisible(False)
        self.browse_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("")
        
    def set_document_processor(self, processor: DocumentProcessor):
        """Set the document processor for uploads."""
        self.document_processor = processor


class DocumentListWidget(QWidget):
    """Widget for displaying document list with search and filters."""
    
    # Signals
    document_selected = Signal(str)  # document_id
    document_deleted = Signal(str)  # document_id
    documents_refreshed = Signal()
    
    def __init__(self):
        super().__init__()
        self.documents: Dict[str, DocumentInfo] = {}
        
        self._setup_ui()
        self._connect_signals()
        self._load_sample_documents()
        
    def _setup_ui(self):
        """Set up the document list UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Header with search and controls
        header_layout = QHBoxLayout()
        
        # Search bar
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search documents...")
        
        # Filter combo
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Documents", "PDF", "Word", "Text", "Processed", "Unprocessed"])
        self.filter_combo.setMaximumWidth(120)
        
        # Refresh button
        self.refresh_button = QToolButton()
        self.refresh_button.setText("🔄")
        self.refresh_button.setToolTip("Refresh")
        
        header_layout.addWidget(self.search_edit, 1)
        header_layout.addWidget(self.filter_combo)
        header_layout.addWidget(self.refresh_button)
        
        layout.addLayout(header_layout)
        
        # Document table
        self.document_table = QTableWidget()
        self.document_table.setColumnCount(6)
        self.document_table.setHorizontalHeaderLabels([
            "Name", "Type", "Size", "Date", "Status", "Chunks"
        ])
        
        # Configure table
        header = self.document_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Name column
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Type
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Size
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Date
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Status
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Chunks
        
        self.document_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.document_table.setContextMenuPolicy(Qt.CustomContextMenu)
        
        layout.addWidget(self.document_table)
        
        # Apply styling
        self._apply_styling()
        
    def _apply_styling(self):
        """Apply styling to the document list."""
        self.setStyleSheet("""
            DocumentListWidget {
                background-color: #2b2b2b;
            }
            
            QLineEdit {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
            }
            
            QLineEdit:focus {
                border-color: #0078d4;
            }
            
            QComboBox {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
            }
            
            QComboBox::drop-down {
                border: none;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #ffffff;
            }
            
            QComboBox QAbstractItemView {
                background-color: #404040;
                border: 1px solid #555555;
                selection-background-color: #0078d4;
                color: #ffffff;
            }
            
            QToolButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
            }
            
            QToolButton:hover {
                background-color: #505050;
            }
            
            QTableWidget {
                background-color: #2b2b2b;
                alternate-background-color: #353535;
                gridline-color: #404040;
                color: #ffffff;
                border: 1px solid #555555;
            }
            
            QTableWidget::item {
                padding: 8px;
                border: none;
            }
            
            QTableWidget::item:selected {
                background-color: #0078d4;
            }
            
            QHeaderView::section {
                background-color: #404040;
                color: #ffffff;
                padding: 8px;
                border: 1px solid #555555;
                font-weight: bold;
            }
        """)
        
    def _connect_signals(self):
        """Connect widget signals."""
        self.search_edit.textChanged.connect(self._filter_documents)
        self.filter_combo.currentTextChanged.connect(self._filter_documents)
        self.refresh_button.clicked.connect(self._refresh_documents)
        self.document_table.itemSelectionChanged.connect(self._on_selection_changed)
        self.document_table.customContextMenuRequested.connect(self._show_context_menu)
        
    def _load_sample_documents(self):
        """Load sample documents for testing."""
        sample_docs = [
            DocumentInfo(
                id="doc_1",
                title="Machine Learning Guide",
                filename="ml_guide.pdf",
                file_type="PDF",
                file_size=2048576,  # 2MB
                created_at=datetime.now(),
                processed=True,
                chunk_count=45,
                metadata={"author": "AI Research Lab", "pages": 120}
            ),
            DocumentInfo(
                id="doc_2", 
                title="Python Best Practices",
                filename="python_practices.docx",
                file_type="DOCX",
                file_size=512000,  # 512KB
                created_at=datetime.now(),
                processed=True,
                chunk_count=23,
                metadata={"author": "Developer Community", "words": 5000}
            ),
            DocumentInfo(
                id="doc_3",
                title="Project Documentation",
                filename="project_docs.txt",
                file_type="TXT",
                file_size=102400,  # 100KB
                created_at=datetime.now(),
                processed=False,
                chunk_count=0,
                metadata={"version": "1.0"}
            )
        ]
        
        for doc in sample_docs:
            self.add_document(doc)
            
    def add_document(self, doc_info: DocumentInfo):
        """Add a document to the list."""
        self.documents[doc_info.id] = doc_info
        self._refresh_table()
        
    def remove_document(self, document_id: str):
        """Remove a document from the list."""
        if document_id in self.documents:
            del self.documents[document_id]
            self._refresh_table()
            
    def _refresh_table(self):
        """Refresh the document table."""
        # Get filtered documents
        filtered_docs = self._get_filtered_documents()
        
        # Update table
        self.document_table.setRowCount(len(filtered_docs))
        
        for row, doc in enumerate(filtered_docs):
            # Name
            name_item = QTableWidgetItem(doc.title)
            name_item.setData(Qt.UserRole, doc.id)
            self.document_table.setItem(row, 0, name_item)
            
            # Type
            type_item = QTableWidgetItem(doc.file_type)
            self.document_table.setItem(row, 1, type_item)
            
            # Size
            size_str = self._format_file_size(doc.file_size)
            size_item = QTableWidgetItem(size_str)
            self.document_table.setItem(row, 2, size_item)
            
            # Date
            date_str = doc.created_at.strftime("%Y-%m-%d %H:%M")
            date_item = QTableWidgetItem(date_str)
            self.document_table.setItem(row, 3, date_item)
            
            # Status
            status = "Processed" if doc.processed else "Pending"
            status_item = QTableWidgetItem(status)
            if doc.processed:
                status_item.setBackground(Qt.darkGreen)
            else:
                status_item.setBackground(Qt.darkYellow)
            self.document_table.setItem(row, 4, status_item)
            
            # Chunks
            chunks_item = QTableWidgetItem(str(doc.chunk_count))
            self.document_table.setItem(row, 5, chunks_item)
            
    def _get_filtered_documents(self) -> List[DocumentInfo]:
        """Get documents matching current filters."""
        search_text = self.search_edit.text().lower()
        filter_type = self.filter_combo.currentText()
        
        filtered = []
        for doc in self.documents.values():
            # Search filter
            if search_text and search_text not in doc.title.lower():
                continue
                
            # Type filter
            if filter_type == "All Documents":
                pass
            elif filter_type == "PDF" and doc.file_type != "PDF":
                continue
            elif filter_type == "Word" and doc.file_type != "DOCX":
                continue
            elif filter_type == "Text" and doc.file_type != "TXT":
                continue
            elif filter_type == "Processed" and not doc.processed:
                continue
            elif filter_type == "Unprocessed" and doc.processed:
                continue
                
            filtered.append(doc)
            
        return sorted(filtered, key=lambda x: x.created_at, reverse=True)
        
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
            
    def _filter_documents(self):
        """Apply current filters to document list."""
        self._refresh_table()
        
    def _refresh_documents(self):
        """Refresh the document list."""
        # TODO: Reload from database
        self._refresh_table()
        self.documents_refreshed.emit()
        
    def _on_selection_changed(self):
        """Handle document selection change."""
        current_row = self.document_table.currentRow()
        if current_row >= 0:
            name_item = self.document_table.item(current_row, 0)
            if name_item:
                document_id = name_item.data(Qt.UserRole)
                self.document_selected.emit(document_id)
                
    def _show_context_menu(self, position):
        """Show context menu for document operations."""
        item = self.document_table.itemAt(position)
        if not item:
            return
            
        document_id = self.document_table.item(item.row(), 0).data(Qt.UserRole)
        doc_info = self.documents.get(document_id)
        if not doc_info:
            return
            
        menu = QMenu(self)
        
        # View details
        view_action = QAction("View Details", self)
        view_action.triggered.connect(lambda: self._view_document_details(document_id))
        menu.addAction(view_action)
        
        # Reprocess (if not processed)
        if not doc_info.processed:
            reprocess_action = QAction("Process Document", self)
            reprocess_action.triggered.connect(lambda: self._reprocess_document(document_id))
            menu.addAction(reprocess_action)
            
        menu.addSeparator()
        
        # Delete
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._delete_document(document_id))
        menu.addAction(delete_action)
        
        menu.exec(self.document_table.mapToGlobal(position))
        
    def _view_document_details(self, document_id: str):
        """View detailed information about a document."""
        # TODO: Implement document details dialog
        QMessageBox.information(self, "Document Details", f"View details for document: {document_id}")
        
    def _reprocess_document(self, document_id: str):
        """Reprocess a document."""
        # TODO: Implement document reprocessing
        QMessageBox.information(self, "Reprocess", f"Reprocessing document: {document_id}")
        
    def _delete_document(self, document_id: str):
        """Delete a document with confirmation."""
        doc_info = self.documents.get(document_id)
        if not doc_info:
            return
            
        reply = QMessageBox.question(
            self,
            "Delete Document",
            f"Are you sure you want to delete '{doc_info.title}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.remove_document(document_id)
            self.document_deleted.emit(document_id)


class DocumentManager(QWidget):
    """
    Main document management widget combining upload and library functionality.
    """
    
    # Signals
    document_uploaded = Signal(str)  # document_id
    document_selected = Signal(str)  # document_id
    
    def __init__(self, event_bus: EventBus):
        super().__init__()
        self.event_bus = event_bus
        self.logger = logger.bind(component="DocumentManager")
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        """Set up the document manager UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Upload tab
        self.upload_widget = DocumentUploadWidget()
        self.tab_widget.addTab(self.upload_widget, "📤 Upload")
        
        # Library tab
        self.library_widget = DocumentListWidget()
        self.tab_widget.addTab(self.library_widget, "📚 Library")
        
        layout.addWidget(self.tab_widget)
        
        # Apply styling
        self._apply_styling()
        
    def _apply_styling(self):
        """Apply styling to the document manager."""
        self.setStyleSheet("""
            DocumentManager {
                background-color: #2b2b2b;
            }
            
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            
            QTabWidget::tab-bar {
                alignment: left;
            }
            
            QTabBar::tab {
                background-color: #404040;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            
            QTabBar::tab:hover:!selected {
                background-color: #505050;
            }
        """)
        
    def _connect_signals(self):
        """Connect widget signals."""
        self.upload_widget.documents_uploaded.connect(self._on_documents_uploaded)
        self.library_widget.document_selected.connect(self.document_selected)
        
    def _on_documents_uploaded(self, file_paths: List[str]):
        """Handle document upload completion."""
        self.logger.info(f"Documents uploaded: {len(file_paths)} files")
        
        # Switch to library tab to show uploaded documents
        self.tab_widget.setCurrentIndex(1)
        
        # Refresh library
        self.library_widget._refresh_documents()
        
    def set_document_processor(self, processor: DocumentProcessor):
        """Set the document processor for uploads."""
        self.upload_widget.set_document_processor(processor)
        
    def get_selected_document_id(self) -> Optional[str]:
        """Get the currently selected document ID."""
        # Implementation would depend on current tab and selection
        return None
        
    def refresh_library(self):
        """Refresh the document library."""
        self.library_widget._refresh_documents()