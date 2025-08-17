"""
Evaluation dashboard component for RAG quality monitoring.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...core.events import EventBus
from ...services.rag_evaluation_service import RAGEvaluationService, ResponseEvaluation


class MetricCard(QFrame):
    """Card widget for displaying a single metric."""
    
    def __init__(self, title: str, value: float, max_value: float = 1.0, format_str: str = "{:.3f}"):
        super().__init__()
        self.title = title
        self.value = value
        self.max_value = max_value
        self.format_str = format_str
        
        self._setup_ui()
        self._apply_styling()
        
    def _setup_ui(self) -> None:
        """Set up the metric card UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        
        # Value
        self.value_label = QLabel(self.format_str.format(self.value))
        self.value_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.value_label.setAlignment(Qt.AlignCenter)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int((self.value / self.max_value) * 100))
        self.progress_bar.setFixedHeight(6)
        
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.progress_bar)
        
    def _apply_styling(self) -> None:
        """Apply styling to the metric card."""
        # Color based on value
        if self.value >= 0.8 * self.max_value:
            color = "#00ff00"  # Green
        elif self.value >= 0.6 * self.max_value:
            color = "#ffaa00"  # Orange
        else:
            color = "#ff4444"  # Red
            
        self.setStyleSheet(f"""
            MetricCard {{
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 8px;
                border-left: 4px solid {color};
            }}
            
            QLabel {{
                color: #ffffff;
                background-color: transparent;
            }}
            
            QProgressBar {{
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #2b2b2b;
            }}
            
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """)
        
    def update_value(self, value: float) -> None:
        """Update the metric value."""
        self.value = value
        self.value_label.setText(self.format_str.format(value))
        self.progress_bar.setValue(int((value / self.max_value) * 100))
        self._apply_styling()  # Update colors


class EvaluationHistoryTable(QTableWidget):
    """Table widget for displaying evaluation history."""
    
    def __init__(self):
        super().__init__()
        self._setup_table()
        
    def _setup_table(self) -> None:
        """Set up the evaluation history table."""
        headers = [
            "Timestamp", "Query", "Overall Score", "Relevance", "Coherence", 
            "Completeness", "Citation Accuracy", "Response Time", "Sources"
        ]
        
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        
        # Set column widths
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Timestamp
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Query
        for i in range(2, len(headers)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
            
        # Styling
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                color: #ffffff;
                gridline-color: #555555;
            }
            
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #404040;
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
        
    def add_evaluation(self, evaluation: ResponseEvaluation) -> None:
        """Add an evaluation to the table."""
        row = self.rowCount()
        self.insertRow(row)
        
        # Format data
        timestamp = evaluation.timestamp.strftime("%H:%M:%S")
        query_preview = evaluation.query[:50] + "..." if len(evaluation.query) > 50 else evaluation.query
        
        # Add items
        items = [
            timestamp,
            query_preview,
            f"{evaluation.overall_score:.3f}",
            f"{evaluation.relevance_score:.3f}",
            f"{evaluation.coherence_score:.3f}",
            f"{evaluation.completeness_score:.3f}",
            f"{evaluation.citation_accuracy:.3f}",
            f"{evaluation.response_time:.2f}s",
            str(evaluation.retrieval_context.total_retrieved)
        ]
        
        for col, item_text in enumerate(items):
            item = QTableWidgetItem(item_text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            
            # Color code score columns
            if col in [2, 3, 4, 5, 6]:  # Score columns
                score = float(item_text)
                if score >= 0.8:
                    item.setBackground(QColor("#004d00"))  # Dark green
                elif score >= 0.6:
                    item.setBackground(QColor("#4d4d00"))  # Dark yellow
                else:
                    item.setBackground(QColor("#4d0000"))  # Dark red
                    
            self.setItem(row, col, item)
            
        # Scroll to bottom
        self.scrollToBottom()


class EvaluationDashboard(QWidget):
    """
    Comprehensive evaluation dashboard for RAG system monitoring.
    """
    
    # Signals
    export_requested = Signal(str)  # Export file path
    
    def __init__(self, event_bus: EventBus, evaluation_service: Optional[RAGEvaluationService] = None):
        super().__init__()
        self.event_bus = event_bus
        self.evaluation_service = evaluation_service
        self.logger = logger.bind(component="EvaluationDashboard")
        
        self.recent_evaluations: List[ResponseEvaluation] = []
        self.metric_cards: Dict[str, MetricCard] = {}
        
        self._setup_ui()
        self._setup_refresh_timer()
        
        if self.evaluation_service:
            self._load_initial_data()
            
    def _setup_ui(self) -> None:
        """Set up the dashboard UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("RAG Evaluation Dashboard")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setFixedSize(80, 32)
        self.refresh_button.clicked.connect(self._refresh_data)
        
        self.export_button = QPushButton("Export Data")
        self.export_button.setFixedSize(100, 32)
        self.export_button.clicked.connect(self._export_data)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.refresh_button)
        header_layout.addWidget(self.export_button)
        
        layout.addLayout(header_layout)
        
        # Main content tabs
        self.tab_widget = QTabWidget()
        
        # Overview tab
        self.overview_tab = self._create_overview_tab()
        self.tab_widget.addTab(self.overview_tab, "Overview")
        
        # History tab
        self.history_tab = self._create_history_tab()
        self.tab_widget.addTab(self.history_tab, "History")
        
        # Details tab
        self.details_tab = self._create_details_tab()
        self.tab_widget.addTab(self.details_tab, "Details")
        
        layout.addWidget(self.tab_widget)
        
        # Apply styling
        self._apply_styling()
        
    def _create_overview_tab(self) -> QWidget:
        """Create the overview tab with key metrics."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        
        # Metrics grid
        metrics_group = QGroupBox("Current Performance Metrics")
        metrics_layout = QGridLayout(metrics_group)
        metrics_layout.setSpacing(12)
        
        # Create metric cards
        metric_configs = [
            ("Overall Score", 0.0, 1.0, "{:.3f}"),
            ("Relevance", 0.0, 1.0, "{:.3f}"),
            ("Coherence", 0.0, 1.0, "{:.3f}"),
            ("Completeness", 0.0, 1.0, "{:.3f}"),
            ("Citation Accuracy", 0.0, 1.0, "{:.3f}"),
            ("Factual Accuracy", 0.0, 1.0, "{:.3f}"),
            ("Avg Response Time", 0.0, 5.0, "{:.2f}s"),
            ("Confidence", 0.0, 1.0, "{:.3f}")
        ]
        
        for i, (title, value, max_val, fmt) in enumerate(metric_configs):
            card = MetricCard(title, value, max_val, fmt)
            self.metric_cards[title.lower().replace(" ", "_")] = card
            metrics_layout.addWidget(card, i // 4, i % 4)
            
        layout.addWidget(metrics_group)
        
        # Performance trend summary
        trend_group = QGroupBox("Performance Trend")
        trend_layout = QVBoxLayout(trend_group)
        
        self.trend_label = QLabel("No evaluation data available")
        self.trend_label.setFont(QFont("Arial", 12))
        self.trend_label.setAlignment(Qt.AlignCenter)
        
        trend_layout.addWidget(self.trend_label)
        layout.addWidget(trend_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout(stats_group)
        
        self.total_evaluations_label = QLabel("0")
        self.avg_retrieval_time_label = QLabel("0.0s")
        self.avg_sources_label = QLabel("0")
        self.success_rate_label = QLabel("0%")
        
        stats_layout.addRow("Total Evaluations:", self.total_evaluations_label)
        stats_layout.addRow("Avg Retrieval Time:", self.avg_retrieval_time_label)
        stats_layout.addRow("Avg Sources Retrieved:", self.avg_sources_label)
        stats_layout.addRow("Success Rate (>0.7):", self.success_rate_label)
        
        layout.addWidget(stats_group)
        layout.addStretch()
        
        return tab
        
    def _create_history_tab(self) -> QWidget:
        """Create the history tab with evaluation timeline."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.history_info_label = QLabel("Recent evaluations")
        self.clear_history_button = QPushButton("Clear History")
        self.clear_history_button.setFixedSize(100, 32)
        self.clear_history_button.clicked.connect(self._clear_history)
        
        controls_layout.addWidget(self.history_info_label)
        controls_layout.addStretch()
        controls_layout.addWidget(self.clear_history_button)
        
        layout.addLayout(controls_layout)
        
        # History table
        self.history_table = EvaluationHistoryTable()
        layout.addWidget(self.history_table)
        
        return tab
        
    def _create_details_tab(self) -> QWidget:
        """Create the details tab with in-depth analysis."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Analysis text area
        details_group = QGroupBox("Detailed Analysis")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setPlainText("Select an evaluation from the history to view detailed analysis.")
        
        details_layout.addWidget(self.details_text)
        layout.addWidget(details_group)
        
        # Configuration info
        config_group = QGroupBox("Evaluation Configuration")
        config_layout = QFormLayout(config_group)
        
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setMaximumHeight(150)
        
        config_layout.addWidget(self.config_text)
        layout.addWidget(config_group)
        
        return tab
        
    def _apply_styling(self) -> None:
        """Apply styling to the dashboard."""
        self.setStyleSheet("""
            EvaluationDashboard {
                background-color: #1e1e1e;
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
            
            QTabBar::tab:hover {
                background-color: #555555;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 4px;
                margin: 8px 0px;
                padding-top: 16px;
                color: #ffffff;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 8px 0 8px;
            }
            
            QLabel {
                color: #ffffff;
            }
            
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 4px;
                color: #ffffff;
                font-weight: bold;
                padding: 4px 8px;
            }
            
            QPushButton:hover {
                background-color: #106ebe;
            }
            
            QPushButton:pressed {
                background-color: #005a9e;
            }
            
            QTextEdit {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #ffffff;
                padding: 8px;
            }
        """)
        
    def _setup_refresh_timer(self) -> None:
        """Set up automatic refresh timer."""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_data)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
        
    def _load_initial_data(self) -> None:
        """Load initial evaluation data."""
        if self.evaluation_service:
            self._refresh_data()
            
    def _refresh_data(self) -> None:
        """Refresh dashboard data."""
        if not self.evaluation_service:
            return
            
        try:
            # Get evaluation summary
            summary = self.evaluation_service.get_evaluation_summary()
            
            # Update metric cards
            avg_scores = summary.get('average_scores', {})
            for metric_name, card in self.metric_cards.items():
                if metric_name in avg_scores:
                    card.update_value(avg_scores[metric_name])
                elif metric_name == "avg_response_time":
                    perf_metrics = summary.get('performance_metrics', {})
                    if 'avg_response_time' in perf_metrics:
                        card.update_value(perf_metrics['avg_response_time'])
                        
            # Update statistics
            self.total_evaluations_label.setText(str(summary.get('total_evaluations', 0)))
            
            perf_metrics = summary.get('performance_metrics', {})
            self.avg_retrieval_time_label.setText(f"{perf_metrics.get('avg_response_time', 0):.2f}s")
            
            # Calculate success rate
            if 'average_scores' in summary and 'overall_score' in summary['average_scores']:
                overall_avg = summary['average_scores']['overall_score']
                success_rate = 100 if overall_avg >= 0.7 else int(overall_avg * 100)
                self.success_rate_label.setText(f"{success_rate}%")
                
            # Update trend
            trend = summary.get('evaluation_trend', 'insufficient_data')
            trend_text = {
                'improving': "📈 Performance is improving",
                'stable': "➡️ Performance is stable", 
                'declining': "📉 Performance is declining",
                'insufficient_data': "📊 Insufficient data for trend analysis"
            }.get(trend, "Unknown trend")
            
            self.trend_label.setText(trend_text)
            
            # Update recent evaluations in history
            recent_evals = self.evaluation_service.evaluation_history[-10:]  # Last 10
            for evaluation in recent_evals:
                if evaluation not in self.recent_evaluations:
                    self.recent_evaluations.append(evaluation)
                    self.history_table.add_evaluation(evaluation)
                    
            # Update info label
            self.history_info_label.setText(f"Showing {len(self.recent_evaluations)} recent evaluations")
            
            self.logger.debug("Dashboard data refreshed")
            
        except Exception as e:
            self.logger.error(f"Failed to refresh dashboard data: {e}")
            
    def _clear_history(self) -> None:
        """Clear evaluation history display."""
        self.recent_evaluations.clear()
        self.history_table.setRowCount(0)
        self.history_info_label.setText("History cleared")
        
    def _export_data(self) -> None:
        """Export evaluation data to file."""
        if not self.evaluation_service:
            return
            
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Evaluation Data",
                f"rag_evaluation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "JSON Files (*.json)"
            )
            
            if file_path:
                self.evaluation_service.export_evaluations(file_path)
                self.export_requested.emit(file_path)
                self.logger.info(f"Evaluation data exported to: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to export evaluation data: {e}")
            
    def set_evaluation_service(self, evaluation_service: RAGEvaluationService) -> None:
        """Set the evaluation service."""
        self.evaluation_service = evaluation_service
        self._load_initial_data()
        
    def add_evaluation(self, evaluation: ResponseEvaluation) -> None:
        """Add a new evaluation to the dashboard."""
        self.recent_evaluations.append(evaluation)
        self.history_table.add_evaluation(evaluation)
        
        # Trigger refresh to update metrics
        self._refresh_data()
        
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        return {name: card.value for name, card in self.metric_cards.items()}