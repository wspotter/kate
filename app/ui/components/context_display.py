"""
Context display UI components for Kate LLM Client.

Provides comprehensive visualization of RAG context including retrieved documents,
conversation snippets, and relevance information for enhanced transparency.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame,
    QLabel, QPushButton, QTextEdit, QGroupBox, QProgressBar,
    QTreeWidget, QTreeWidgetItem, QSplitter, QTabWidget,
    QCollapsibleGroupBox, QToolButton, QMenu, QAction,
    QHeaderView, QAbstractItemView, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QCursor
from loguru import logger

from ...core.events import EventBus
from ...services.rag_service import ContextSource


@dataclass
class ContextItem:
    """Represents a context item for display."""
    source_id: str
    source_type: str  # 'document', 'conversation'
    title: str
    content: str
    snippet: str
    score: float
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    citation: Optional[str] = None


class CollapsibleFrame(QFrame):
    """A collapsible frame widget."""
    
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.is_expanded = True
        
        self._setup_ui()
        self._setup_animation()
        
    def _setup_ui(self):
        """Set up the collapsible frame UI."""
        self.setFrameStyle(QFrame.StyledPanel)
        self.setLineWidth(1)
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Header
        self.header = QFrame()
        self.header.setFixedHeight(30)
        self.header.setStyleSheet("""
            QFrame {
                background-color: #404040;
                border: none;
                border-radius: 3px;
            }
            QFrame:hover {
                background-color: #4a4a4a;
            }
        """)
        
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        
        # Toggle button
        self.toggle_button = QToolButton()
        self.toggle_button.setText("▼")
        self.toggle_button.setFixedSize(20, 20)
        self.toggle_button.clicked.connect(self.toggle_expanded)
        
        # Title label
        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont("Arial", 9, QFont.Bold))
        
        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        
        # Content area
        self.content_area = QFrame()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        
        self.main_layout.addWidget(self.header)
        self.main_layout.addWidget(self.content_area)
        
    def _setup_animation(self):
        """Set up collapse/expand animation."""
        self.animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        
    def add_widget(self, widget: QWidget):
        """Add widget to the content area."""
        self.content_layout.addWidget(widget)
        
    def toggle_expanded(self):
        """Toggle the expanded state."""
        self.set_expanded(not self.is_expanded)
        
    def set_expanded(self, expanded: bool):
        """Set the expanded state."""
        self.is_expanded = expanded
        
        if expanded:
            self.toggle_button.setText("▼")
            # Expand animation
            self.content_area.setMaximumHeight(0)
            self.content_area.show()
            self.animation.setStartValue(0)
            self.animation.setEndValue(self.content_area.sizeHint().height())
        else:
            self.toggle_button.setText("▶")
            # Collapse animation
            self.animation.setStartValue(self.content_area.height())
            self.animation.setEndValue(0)
            
        self.animation.finished.connect(
            lambda: self.content_area.hide() if not expanded else None
        )
        self.animation.start()


class ContextSourceWidget(QFrame):
    """Widget for displaying a single context source."""
    
    # Signals
    source_clicked = Signal(str)  # source_id
    citation_requested = Signal(str)  # source_id
    
    def __init__(self, context_item: ContextItem, parent=None):
        super().__init__(parent)
        self.context_item = context_item
        
        self._setup_ui()
        self._apply_styling()
        
    def _setup_ui(self):
        """Set up the context source widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        # Header with title and metadata
        header_layout = QHBoxLayout()
        
        # Source type icon
        type_icon = "📄" if self.context_item.source_type == "document" else "💬"
        icon_label = QLabel(type_icon)
        icon_label.setFont(QFont("Arial", 16))
        
        # Title and metadata
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        # Title
        self.title_label = QLabel(self.context_item.title)
        self.title_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.title_label.setWordWrap(True)
        self.title_label.setMaximumHeight(40)
        
        # Metadata line
        metadata_parts = []
        if self.context_item.created_at:
            metadata_parts.append(self.context_item.created_at.strftime("%Y-%m-%d"))
        if self.context_item.metadata:
            if "file_type" in self.context_item.metadata:
                metadata_parts.append(self.context_item.metadata["file_type"].upper())
        
        metadata_text = " • ".join(metadata_parts)
        self.metadata_label = QLabel(metadata_text)
        self.metadata_label.setFont(QFont("Arial", 8))
        self.metadata_label.setStyleSheet("color: #888888;")
        
        info_layout.addWidget(self.title_label)
        info_layout.addWidget(self.metadata_label)
        
        # Score and actions
        actions_layout = QVBoxLayout()
        actions_layout.setAlignment(Qt.AlignRight | Qt.AlignTop)
        
        # Relevance score
        score_text = f"{self.context_item.score:.1%}"
        self.score_label = QLabel(score_text)
        self.score_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.score_label.setAlignment(Qt.AlignRight)
        
        # Set score color based on value
        if self.context_item.score >= 0.8:
            color = "#4CAF50"  # Green
        elif self.context_item.score >= 0.6:
            color = "#FF9800"  # Orange
        else:
            color = "#F44336"  # Red
        self.score_label.setStyleSheet(f"color: {color};")
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(4)
        
        # View source button
        self.view_button = QPushButton("View")
        self.view_button.setFixedSize(40, 20)
        self.view_button.clicked.connect(lambda: self.source_clicked.emit(self.context_item.source_id))
        
        # Citation button
        if self.context_item.citation:
            self.cite_button = QPushButton("Cite")
            self.cite_button.setFixedSize(40, 20)
            self.cite_button.clicked.connect(lambda: self.citation_requested.emit(self.context_item.source_id))
            button_layout.addWidget(self.cite_button)
            
        button_layout.addWidget(self.view_button)
        
        actions_layout.addWidget(self.score_label)
        actions_layout.addLayout(button_layout)
        
        header_layout.addWidget(icon_label)
        header_layout.addLayout(info_layout, 1)
        header_layout.addLayout(actions_layout)
        
        layout.addLayout(header_layout)
        
        # Content snippet
        self.content_text = QTextEdit()
        self.content_text.setPlainText(self.context_item.snippet)
        self.content_text.setReadOnly(True)
        self.content_text.setMaximumHeight(80)
        self.content_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.content_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        layout.addWidget(self.content_text)
        
        # Citation info
        if self.context_item.citation:
            citation_label = QLabel(f"Citation: {self.context_item.citation}")
            citation_label.setFont(QFont("Arial", 8))
            citation_label.setStyleSheet("color: #666666; font-style: italic;")
            layout.addWidget(citation_label)
            
    def _apply_styling(self):
        """Apply styling to the context source widget."""
        self.setStyleSheet("""
            ContextSourceWidget {
                background-color: #353535;
                border: 1px solid #555555;
                border-radius: 6px;
                margin: 2px;
            }
            
            ContextSourceWidget:hover {
                border-color: #0078d4;
                background-color: #3a3a3a;
            }
            
            QLabel {
                color: #ffffff;
                border: none;
                background: transparent;
            }
            
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 3px;
                color: white;
                font-size: 8px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #106ebe;
            }
            
            QPushButton:pressed {
                background-color: #005a9e;
            }
            
            QTextEdit {
                background-color: #2b2b2b;
                border: 1px solid #444444;
                border-radius: 3px;
                color: #ffffff;
                font-family: "Consolas", "Monaco", monospace;
                font-size: 9px;
                padding: 4px;
            }
        """)


class ContextDisplayWidget(QWidget):
    """Main widget for displaying RAG context information."""
    
    # Signals
    source_selected = Signal(str, str)  # source_id, source_type
    context_cleared = Signal()
    
    def __init__(self, event_bus: EventBus, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.logger = logger.bind(component="ContextDisplayWidget")
        
        # Context data
        self.context_sources: List[ContextItem] = []
        self.current_query: str = ""
        self.generation_time: float = 0.0
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        """Set up the context display UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header_frame = QFrame()
        header_frame.setFixedHeight(40)
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #404040;
                border-bottom: 1px solid #555555;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 8, 12, 8)
        
        # Title
        title_label = QLabel("Context Sources")
        title_label.setFont(QFont("Arial", 11, QFont.Bold))
        title_label.setStyleSheet("color: #ffffff;")
        
        # Info label
        self.info_label = QLabel("No context available")
        self.info_label.setFont(QFont("Arial", 9))
        self.info_label.setStyleSheet("color: #888888;")
        
        # Clear button
        self.clear_button = QPushButton("Clear")
        self.clear_button.setFixedSize(60, 24)
        self.clear_button.clicked.connect(self.clear_context)
        self.clear_button.setEnabled(False)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(self.info_label, 1)
        header_layout.addWidget(self.clear_button)
        
        # Content area with tabs
        self.tab_widget = QTabWidget()
        
        # Sources tab
        self.sources_tab = QWidget()
        self._setup_sources_tab()
        self.tab_widget.addTab(self.sources_tab, "Sources")
        
        # Summary tab
        self.summary_tab = QWidget()
        self._setup_summary_tab()
        self.tab_widget.addTab(self.summary_tab, "Summary")
        
        layout.addWidget(header_frame)
        layout.addWidget(self.tab_widget)
        
        # Apply global styling
        self._apply_styling()
        
    def _setup_sources_tab(self):
        """Set up the sources tab."""
        layout = QVBoxLayout(self.sources_tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Scroll area for context sources
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Container for context widgets
        self.sources_container = QWidget()
        self.sources_layout = QVBoxLayout(self.sources_container)
        self.sources_layout.setContentsMargins(0, 0, 0, 0)
        self.sources_layout.setSpacing(6)
        self.sources_layout.addStretch()
        
        self.scroll_area.setWidget(self.sources_container)
        layout.addWidget(self.scroll_area)
        
    def _setup_summary_tab(self):
        """Set up the summary tab."""
        layout = QVBoxLayout(self.summary_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Query info
        query_group = QGroupBox("Current Query")
        query_layout = QVBoxLayout(query_group)
        
        self.query_label = QLabel("No query")
        self.query_label.setWordWrap(True)
        self.query_label.setStyleSheet("color: #ffffff; padding: 8px;")
        query_layout.addWidget(self.query_label)
        
        # Context statistics
        stats_group = QGroupBox("Context Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(120)
        self.stats_text.setPlainText("No statistics available")
        stats_layout.addWidget(self.stats_text)
        
        # Source breakdown
        breakdown_group = QGroupBox("Source Breakdown")
        breakdown_layout = QVBoxLayout(breakdown_group)
        
        self.breakdown_text = QTextEdit()
        self.breakdown_text.setReadOnly(True)
        self.breakdown_text.setMaximumHeight(100)
        self.breakdown_text.setPlainText("No sources")
        breakdown_layout.addWidget(self.breakdown_text)
        
        layout.addWidget(query_group)
        layout.addWidget(stats_group)
        layout.addWidget(breakdown_group)
        layout.addStretch()
        
    def _apply_styling(self):
        """Apply styling to the context display."""
        self.setStyleSheet("""
            ContextDisplayWidget {
                background-color: #2b2b2b;
            }
            
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            
            QTabBar::tab {
                background-color: #404040;
                color: #ffffff;
                padding: 6px 12px;
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
            
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 4px;
                color: white;
                font-weight: bold;
                padding: 4px 8px;
            }
            
            QPushButton:hover {
                background-color: #106ebe;
            }
            
            QPushButton:pressed {
                background-color: #005a9e;
            }
            
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 12px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 6px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
            
            QTextEdit {
                background-color: #353535;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #ffffff;
                padding: 8px;
            }
        """)
        
    def _connect_signals(self):
        """Connect widget signals."""
        pass  # Will be connected externally
        
    def set_context(
        self, 
        sources: List[ContextSource], 
        query: str = "",
        generation_time: float = 0.0
    ):
        """
        Set the context sources to display.
        
        Args:
            sources: List of context sources
            query: The query that generated this context
            generation_time: Time taken to generate response
        """
        self.current_query = query
        self.generation_time = generation_time
        
        # Convert ContextSource to ContextItem
        self.context_sources = []
        for source in sources:
            context_item = ContextItem(
                source_id=source.title or "unknown",  # Use title as ID for now
                source_type=source.source_type,
                title=source.title or "Untitled",
                content=source.content,
                snippet=source.content[:200] + "..." if len(source.content) > 200 else source.content,
                score=source.score,
                metadata=source.metadata,
                citation=source.citation
            )
            self.context_sources.append(context_item)
            
        self._update_display()
        
    def _update_display(self):
        """Update the display with current context sources."""
        # Clear existing widgets
        self._clear_sources_layout()
        
        # Update info label
        if self.context_sources:
            self.info_label.setText(f"{len(self.context_sources)} sources")
            self.clear_button.setEnabled(True)
        else:
            self.info_label.setText("No context available")
            self.clear_button.setEnabled(False)
            
        # Add source widgets
        for context_item in self.context_sources:
            source_widget = ContextSourceWidget(context_item)
            source_widget.source_clicked.connect(self._on_source_clicked)
            source_widget.citation_requested.connect(self._on_citation_requested)
            
            # Insert before the stretch
            self.sources_layout.insertWidget(
                self.sources_layout.count() - 1, 
                source_widget
            )
            
        # Update summary tab
        self._update_summary()
        
    def _clear_sources_layout(self):
        """Clear all source widgets from the layout."""
        while self.sources_layout.count() > 1:  # Keep the stretch item
            child = self.sources_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
    def _update_summary(self):
        """Update the summary tab with current context information."""
        # Update query
        self.query_label.setText(self.current_query or "No query")
        
        # Update statistics
        if self.context_sources:
            doc_sources = [s for s in self.context_sources if s.source_type == "document"]
            conv_sources = [s for s in self.context_sources if s.source_type == "conversation"]
            
            avg_score = sum(s.score for s in self.context_sources) / len(self.context_sources)
            max_score = max(s.score for s in self.context_sources)
            min_score = min(s.score for s in self.context_sources)
            
            stats_text = f"""Total Sources: {len(self.context_sources)}
Document Sources: {len(doc_sources)}
Conversation Sources: {len(conv_sources)}

Relevance Scores:
  Average: {avg_score:.1%}
  Highest: {max_score:.1%}
  Lowest: {min_score:.1%}

Generation Time: {self.generation_time:.2f}s"""
            
            self.stats_text.setPlainText(stats_text)
            
            # Update breakdown
            breakdown_lines = []
            for source in self.context_sources[:5]:  # Top 5 sources
                breakdown_lines.append(
                    f"• {source.title[:30]}... ({source.score:.1%})"
                )
            
            if len(self.context_sources) > 5:
                breakdown_lines.append(f"... and {len(self.context_sources) - 5} more")
                
            self.breakdown_text.setPlainText("\n".join(breakdown_lines))
        else:
            self.stats_text.setPlainText("No statistics available")
            self.breakdown_text.setPlainText("No sources")
            
    def _on_source_clicked(self, source_id: str):
        """Handle source click."""
        # Find the source
        source_item = None
        for item in self.context_sources:
            if item.source_id == source_id:
                source_item = item
                break
                
        if source_item:
            self.source_selected.emit(source_item.source_id, source_item.source_type)
            
    def _on_citation_requested(self, source_id: str):
        """Handle citation request."""
        # Find the source and copy citation to clipboard
        for item in self.context_sources:
            if item.source_id == source_id and item.citation:
                from PySide6.QtGui import QGuiApplication
                clipboard = QGuiApplication.clipboard()
                clipboard.setText(item.citation)
                self.logger.info(f"Citation copied to clipboard: {item.citation}")
                break
                
    def clear_context(self):
        """Clear all context sources."""
        self.context_sources.clear()
        self.current_query = ""
        self.generation_time = 0.0
        
        self._update_display()
        self.context_cleared.emit()
        
    def highlight_source(self, source_id: str):
        """Highlight a specific source."""
        # Find and highlight the source widget
        for i in range(self.sources_layout.count() - 1):  # Exclude stretch
            widget = self.sources_layout.itemAt(i).widget()
            if isinstance(widget, ContextSourceWidget):
                if widget.context_item.source_id == source_id:
                    # Temporarily change styling to highlight
                    original_style = widget.styleSheet()
                    widget.setStyleSheet(original_style + """
                        ContextSourceWidget {
                            border: 2px solid #FFD700;
                            background-color: #4a4a00;
                        }
                    """)
                    
                    # Scroll to the widget
                    self.scroll_area.ensureWidgetVisible(widget)
                    
                    # Reset after 2 seconds
                    QTimer.singleShot(2000, lambda: widget.setStyleSheet(original_style))
                    break
                    
    def filter_sources(self, source_type: Optional[str] = None, min_score: float = 0.0):
        """Filter displayed sources by type and score."""
        # This would filter the displayed sources
        # For now, just update visibility of existing widgets
        for i in range(self.sources_layout.count() - 1):
            widget = self.sources_layout.itemAt(i).widget()
            if isinstance(widget, ContextSourceWidget):
                item = widget.context_item
                
                # Check filters
                show = True
                if source_type and item.source_type != source_type:
                    show = False
                if item.score < min_score:
                    show = False
                    
                widget.setVisible(show)
                
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context."""
        return {
            "query": self.current_query,
            "sources_count": len(self.context_sources),
            "document_sources": len([s for s in self.context_sources if s.source_type == "document"]),
            "conversation_sources": len([s for s in self.context_sources if s.source_type == "conversation"]),
            "average_score": sum(s.score for s in self.context_sources) / len(self.context_sources) if self.context_sources else 0.0,
            "generation_time": self.generation_time
        }