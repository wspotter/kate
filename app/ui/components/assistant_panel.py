"""
Assistant panel component for Kate LLM Client.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...core.events import EventBus
from ...services.rag_evaluation_service import RAGEvaluationService, ResponseEvaluation


class AssistantCard(QFrame):
    """Widget for displaying assistant information."""
    
    def __init__(self, assistant_id: str, name: str, description: str, avatar: str = "🤖"):
        super().__init__()
        self.assistant_id = assistant_id
        self.name = name
        self.description = description
        self.avatar = avatar
        
        self._setup_ui()
        self._apply_styling()
        
    def _setup_ui(self) -> None:
        """Set up the assistant card UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Header with avatar and name
        header_layout = QHBoxLayout()
        
        # Avatar
        avatar_label = QLabel(self.avatar)
        avatar_label.setFont(QFont("Arial", 20))
        avatar_label.setFixedSize(40, 40)
        avatar_label.setAlignment(Qt.AlignCenter)
        
        # Name
        name_label = QLabel(self.name)
        name_label.setFont(QFont("Arial", 12, QFont.Bold))
        name_label.setWordWrap(True)
        
        header_layout.addWidget(avatar_label)
        header_layout.addWidget(name_label, 1)
        
        # Description
        desc_label = QLabel(self.description)
        desc_label.setFont(QFont("Arial", 10))
        desc_label.setWordWrap(True)
        desc_label.setMaximumHeight(60)
        
        layout.addLayout(header_layout)
        layout.addWidget(desc_label)
        
    def _apply_styling(self) -> None:
        """Apply styling to the assistant card."""
        self.setStyleSheet("""
            AssistantCard {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 8px;
            }
            
            AssistantCard:hover {
                border-color: #0078d4;
                background-color: #4a4a4a;
            }
            
            QLabel {
                color: #ffffff;
                background-color: transparent;
                border: none;
            }
        """)
        
    def mousePressEvent(self, event) -> None:
        """Handle mouse press for selection."""
        if event.button() == Qt.LeftButton:
            # Emit selection signal through parent
            parent = self.parent()
            while parent and not hasattr(parent, 'assistant_selected'):
                parent = parent.parent()
            if parent and hasattr(parent, 'assistant_selected'):
                parent.assistant_selected.emit(self.assistant_id)
        super().mousePressEvent(event)


class ModelSettingsWidget(QWidget):
    """Widget for model parameter settings."""
    
    # Signals
    settings_changed = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.settings = {
            'temperature': 0.7,
            'max_tokens': 2048,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'stream': True
        }
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self) -> None:
        """Set up the settings UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Temperature
        temp_group = QGroupBox("Temperature")
        temp_layout = QFormLayout(temp_group)
        
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 200)  # 0.0 to 2.0
        self.temp_slider.setValue(int(self.settings['temperature'] * 100))
        
        self.temp_label = QLabel(f"{self.settings['temperature']:.1f}")
        
        temp_row = QHBoxLayout()
        temp_row.addWidget(self.temp_slider, 1)
        temp_row.addWidget(self.temp_label)
        
        temp_layout.addRow(temp_row)
        layout.addWidget(temp_group)
        
        # Max Tokens
        tokens_group = QGroupBox("Max Tokens")
        tokens_layout = QFormLayout(tokens_group)
        
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(1, 8192)
        self.tokens_spin.setValue(self.settings['max_tokens'])
        
        tokens_layout.addRow(self.tokens_spin)
        layout.addWidget(tokens_group)
        
        # Top P
        top_p_group = QGroupBox("Top P")
        top_p_layout = QFormLayout(top_p_group)
        
        self.top_p_slider = QSlider(Qt.Horizontal)
        self.top_p_slider.setRange(0, 100)  # 0.0 to 1.0
        self.top_p_slider.setValue(int(self.settings['top_p'] * 100))
        
        self.top_p_label = QLabel(f"{self.settings['top_p']:.2f}")
        
        top_p_row = QHBoxLayout()
        top_p_row.addWidget(self.top_p_slider, 1)
        top_p_row.addWidget(self.top_p_label)
        
        top_p_layout.addRow(top_p_row)
        layout.addWidget(top_p_group)
        
        # Stream toggle
        self.stream_checkbox = QCheckBox("Enable Streaming")
        self.stream_checkbox.setChecked(self.settings['stream'])
        layout.addWidget(self.stream_checkbox)
        
        layout.addStretch()
        
        # Apply styling
        self._apply_styling()
        
    def _apply_styling(self) -> None:
        """Apply styling to the settings widget."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 4px;
                margin: 8px 0px;
                padding-top: 8px;
                color: #ffffff;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 8px 0 8px;
            }
            
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 6px;
                background: #404040;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #005a9e;
                width: 14px;
                border-radius: 7px;
                margin: -4px 0;
            }
            
            QSlider::handle:horizontal:hover {
                background: #106ebe;
            }
            
            QSpinBox {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
                color: #ffffff;
            }
            
            QCheckBox {
                color: #ffffff;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #404040;
            }
            
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #005a9e;
            }
            
            QLabel {
                color: #ffffff;
                min-width: 40px;
            }
        """)
        
    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self.temp_slider.valueChanged.connect(self._update_temperature)
        self.tokens_spin.valueChanged.connect(self._update_max_tokens)
        self.top_p_slider.valueChanged.connect(self._update_top_p)
        self.stream_checkbox.toggled.connect(self._update_stream)
        
    def _update_temperature(self, value: int) -> None:
        """Update temperature setting."""
        temp = value / 100.0
        self.settings['temperature'] = temp
        self.temp_label.setText(f"{temp:.1f}")
        self.settings_changed.emit(self.settings.copy())
        
    def _update_max_tokens(self, value: int) -> None:
        """Update max tokens setting."""
        self.settings['max_tokens'] = value
        self.settings_changed.emit(self.settings.copy())
        
    def _update_top_p(self, value: int) -> None:
        """Update top_p setting."""
        top_p = value / 100.0
        self.settings['top_p'] = top_p
        self.top_p_label.setText(f"{top_p:.2f}")
        self.settings_changed.emit(self.settings.copy())
        
    def _update_stream(self, checked: bool) -> None:
        """Update streaming setting."""
        self.settings['stream'] = checked
        self.settings_changed.emit(self.settings.copy())
        
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings."""
        return self.settings.copy()
        
    def set_settings(self, settings: Dict[str, Any]) -> None:
        """Set settings values."""
        self.settings.update(settings)
        
        # Update UI elements
        self.temp_slider.setValue(int(self.settings['temperature'] * 100))
        self.tokens_spin.setValue(self.settings['max_tokens'])
        self.top_p_slider.setValue(int(self.settings['top_p'] * 100))
        self.stream_checkbox.setChecked(self.settings['stream'])


class EvaluationMetricsWidget(QWidget):
    """Widget for displaying real-time evaluation metrics."""
    
    def __init__(self):
        super().__init__()
        self.current_evaluation: Optional[ResponseEvaluation] = None
        self._setup_ui()
        
    def _setup_ui(self) -> None:
        """Set up the evaluation metrics UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Overall score indicator
        overall_group = QGroupBox("Response Quality")
        overall_layout = QVBoxLayout(overall_group)
        
        self.overall_score_label = QLabel("--")
        self.overall_score_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.overall_score_label.setAlignment(Qt.AlignCenter)
        
        self.overall_score_bar = QProgressBar()
        self.overall_score_bar.setRange(0, 100)
        self.overall_score_bar.setValue(0)
        self.overall_score_bar.setFixedHeight(8)
        
        overall_layout.addWidget(self.overall_score_label)
        overall_layout.addWidget(self.overall_score_bar)
        
        layout.addWidget(overall_group)
        
        # Individual metrics
        metrics_group = QGroupBox("Metrics Breakdown")
        metrics_layout = QFormLayout(metrics_group)
        
        self.relevance_label = QLabel("--")
        self.coherence_label = QLabel("--")
        self.completeness_label = QLabel("--")
        self.citation_label = QLabel("--")
        self.confidence_label = QLabel("--")
        
        metrics_layout.addRow("Relevance:", self.relevance_label)
        metrics_layout.addRow("Coherence:", self.coherence_label)
        metrics_layout.addRow("Completeness:", self.completeness_label)
        metrics_layout.addRow("Citations:", self.citation_label)
        metrics_layout.addRow("Confidence:", self.confidence_label)
        
        layout.addWidget(metrics_group)
        
        # Performance indicators
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout(perf_group)
        
        self.response_time_label = QLabel("--")
        self.sources_count_label = QLabel("--")
        self.retrieval_time_label = QLabel("--")
        
        perf_layout.addRow("Response Time:", self.response_time_label)
        perf_layout.addRow("Sources Used:", self.sources_count_label)
        perf_layout.addRow("Retrieval Time:", self.retrieval_time_label)
        
        layout.addWidget(perf_group)
        
        # View details button
        self.details_button = QPushButton("View Detailed Analysis")
        self.details_button.setEnabled(False)
        layout.addWidget(self.details_button)
        
        # Apply styling
        self._apply_styling()
        
    def _apply_styling(self) -> None:
        """Apply styling to the evaluation metrics widget."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 4px;
                margin: 4px 0px;
                padding-top: 8px;
                color: #ffffff;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
            }
            
            QLabel {
                color: #ffffff;
                padding: 2px;
            }
            
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #2b2b2b;
            }
            
            QProgressBar::chunk {
                border-radius: 2px;
            }
            
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 4px;
                color: #ffffff;
                font-weight: bold;
                padding: 6px 8px;
            }
            
            QPushButton:hover {
                background-color: #106ebe;
            }
            
            QPushButton:disabled {
                background-color: #404040;
                color: #888888;
            }
        """)
        
    def update_evaluation(self, evaluation: ResponseEvaluation) -> None:
        """Update the display with new evaluation data."""
        self.current_evaluation = evaluation
        
        # Update overall score
        score_text = f"{evaluation.overall_score:.3f}"
        self.overall_score_label.setText(score_text)
        self.overall_score_bar.setValue(int(evaluation.overall_score * 100))
        
        # Color code the overall score
        if evaluation.overall_score >= 0.8:
            color = "#00ff00"  # Green
            self.overall_score_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
        elif evaluation.overall_score >= 0.6:
            color = "#ffaa00"  # Orange
            self.overall_score_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
        else:
            color = "#ff4444"  # Red
            self.overall_score_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
            
        # Update individual metrics
        self.relevance_label.setText(f"{evaluation.relevance_score:.3f}")
        self.coherence_label.setText(f"{evaluation.coherence_score:.3f}")
        self.completeness_label.setText(f"{evaluation.completeness_score:.3f}")
        self.citation_label.setText(f"{evaluation.citation_accuracy:.3f}")
        self.confidence_label.setText(f"{evaluation.confidence_score:.3f}")
        
        # Update performance metrics
        self.response_time_label.setText(f"{evaluation.response_time:.2f}s")
        self.sources_count_label.setText(str(evaluation.retrieval_context.total_retrieved))
        self.retrieval_time_label.setText(f"{evaluation.retrieval_context.retrieval_time:.2f}s")
        
        # Enable details button
        self.details_button.setEnabled(True)
        
    def clear_evaluation(self) -> None:
        """Clear the evaluation display."""
        self.current_evaluation = None
        
        # Reset all labels
        labels = [
            self.overall_score_label, self.relevance_label, self.coherence_label,
            self.completeness_label, self.citation_label, self.confidence_label,
            self.response_time_label, self.sources_count_label, self.retrieval_time_label
        ]
        
        for label in labels:
            label.setText("--")
            
        self.overall_score_bar.setValue(0)
        self.details_button.setEnabled(False)


class AssistantPanel(QWidget):
    """
    Right panel for assistant selection and model settings.
    """
    
    # Signals
    assistant_changed = Signal(str)  # assistant_id
    assistant_selected = Signal(str)  # assistant_id
    model_settings_changed = Signal(dict)
    evaluation_details_requested = Signal(object)  # ResponseEvaluation
    
    def __init__(self, event_bus: EventBus, evaluation_service: Optional[RAGEvaluationService] = None):
        super().__init__()
        self.event_bus = event_bus
        self.evaluation_service = evaluation_service
        self.logger = logger.bind(component="AssistantPanel")
        
        self.assistants: Dict[str, Dict[str, Any]] = {}
        self.current_assistant_id: Optional[str] = None
        
        self._setup_ui()
        self._connect_signals()
        self._load_assistants()
        
    def _setup_ui(self) -> None:
        """Set up the assistant panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        
        # Title
        title_label = QLabel("Assistant")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # Assistant selection dropdown
        self.assistant_combo = QComboBox()
        self.assistant_combo.setMinimumHeight(36)
        layout.addWidget(self.assistant_combo)
        
        # Current assistant card
        self.current_assistant_frame = QFrame()
        self.current_assistant_layout = QVBoxLayout(self.current_assistant_frame)
        self.current_assistant_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.current_assistant_frame)
        
        # Model settings
        settings_label = QLabel("Model Settings")
        settings_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(settings_label)
        
        # Settings scroll area
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.model_settings = ModelSettingsWidget()
        settings_scroll.setWidget(self.model_settings)
        
        layout.addWidget(settings_scroll)
        
        # Evaluation metrics section
        eval_label = QLabel("Response Evaluation")
        eval_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(eval_label)
        
        self.evaluation_widget = EvaluationMetricsWidget()
        layout.addWidget(self.evaluation_widget)
        
        layout.addStretch()
        
        # Apply styling
        self._apply_styling()
        
    def _apply_styling(self) -> None:
        """Apply styling to the assistant panel."""
        self.setStyleSheet("""
            AssistantPanel {
                background-color: #2b2b2b;
                border-left: 1px solid #404040;
            }
            
            QLabel {
                color: #ffffff;
            }
            
            QComboBox {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px;
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
        """)
        
    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.assistant_combo.currentTextChanged.connect(self._on_assistant_changed)
        self.model_settings.settings_changed.connect(self._on_settings_changed)
        self.evaluation_widget.details_button.clicked.connect(self._on_evaluation_details_requested)
        
    def _load_assistants(self) -> None:
        """Load available assistants."""
        # Sample assistants for testing
        sample_assistants = {
            "general": {
                "name": "General Assistant",
                "description": "A helpful general-purpose AI assistant for various tasks and questions.",
                "avatar": "🤖",
                "provider": "openai",
                "model": "gpt-4"
            },
            "coding": {
                "name": "Code Helper",
                "description": "Specialized in programming, debugging, and software development assistance.",
                "avatar": "💻",
                "provider": "openai", 
                "model": "gpt-4"
            },
            "creative": {
                "name": "Creative Writer",
                "description": "Assists with creative writing, storytelling, and content creation.",
                "avatar": "✍️",
                "provider": "anthropic",
                "model": "claude-3-sonnet"
            },
            "analyst": {
                "name": "Data Analyst",
                "description": "Helps with data analysis, statistics, and research tasks.",
                "avatar": "📊",
                "provider": "openai",
                "model": "gpt-4"
            }
        }
        
        self.assistants = sample_assistants
        
        # Populate combo box
        self.assistant_combo.clear()
        for assistant_id, assistant_data in self.assistants.items():
            self.assistant_combo.addItem(assistant_data["name"], assistant_id)
            
        # Select first assistant
        if self.assistants:
            first_id = list(self.assistants.keys())[0]
            self._select_assistant(first_id)
            
    def _select_assistant(self, assistant_id: str) -> None:
        """Select an assistant."""
        if assistant_id not in self.assistants:
            return
            
        self.current_assistant_id = assistant_id
        assistant_data = self.assistants[assistant_id]
        
        # Update combo box
        for i in range(self.assistant_combo.count()):
            if self.assistant_combo.itemData(i) == assistant_id:
                self.assistant_combo.setCurrentIndex(i)
                break
                
        # Update assistant card
        self._update_assistant_card(assistant_data)
        
        self.logger.debug(f"Selected assistant: {assistant_id}")
        
    def _update_assistant_card(self, assistant_data: Dict[str, Any]) -> None:
        """Update the current assistant display card."""
        # Clear existing card
        for i in reversed(range(self.current_assistant_layout.count())):
            child = self.current_assistant_layout.takeAt(i)
            if child.widget():
                child.widget().deleteLater()
                
        # Create new card
        card = AssistantCard(
            self.current_assistant_id,
            assistant_data["name"],
            assistant_data["description"],
            assistant_data["avatar"]
        )
        
        self.current_assistant_layout.addWidget(card)
        
    def _on_assistant_changed(self, assistant_name: str) -> None:
        """Handle assistant selection change."""
        # Find assistant by name
        for assistant_id, assistant_data in self.assistants.items():
            if assistant_data["name"] == assistant_name:
                self._select_assistant(assistant_id)
                self.assistant_changed.emit(assistant_id)
                break
                
    def _on_settings_changed(self, settings: Dict[str, Any]) -> None:
        """Handle model settings change."""
        self.model_settings_changed.emit(settings)
        
    def _on_evaluation_details_requested(self) -> None:
        """Handle evaluation details request."""
        if self.evaluation_widget.current_evaluation:
            self.evaluation_details_requested.emit(self.evaluation_widget.current_evaluation)
        
    def get_current_assistant_id(self) -> Optional[str]:
        """Get the currently selected assistant ID."""
        return self.current_assistant_id
        
    def get_model_settings(self) -> Dict[str, Any]:
        """Get current model settings."""
        return self.model_settings.get_settings()
        
    def add_assistant(self, assistant_id: str, assistant_data: Dict[str, Any]) -> None:
        """Add a new assistant."""
        self.assistants[assistant_id] = assistant_data
        self.assistant_combo.addItem(assistant_data["name"], assistant_id)
        
    def remove_assistant(self, assistant_id: str) -> None:
        """Remove an assistant."""
        if assistant_id in self.assistants:
            del self.assistants[assistant_id]
            
            # Remove from combo box
            for i in range(self.assistant_combo.count()):
                if self.assistant_combo.itemData(i) == assistant_id:
                    self.assistant_combo.removeItem(i)
                    break
                    
            # Select different assistant if this was selected
            if self.current_assistant_id == assistant_id:
                if self.assistants:
                    first_id = list(self.assistants.keys())[0]
                    self._select_assistant(first_id)
                else:
                    self.current_assistant_id = None
                    
    def update_evaluation(self, evaluation: ResponseEvaluation) -> None:
        """Update the evaluation metrics display."""
        self.evaluation_widget.update_evaluation(evaluation)
        
    def clear_evaluation(self) -> None:
        """Clear the evaluation metrics display."""
        self.evaluation_widget.clear_evaluation()
        
    def set_evaluation_service(self, evaluation_service: RAGEvaluationService) -> None:
        """Set the evaluation service."""
        self.evaluation_service = evaluation_service