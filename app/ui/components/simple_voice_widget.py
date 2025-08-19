"""
Simple Voice Widget - 3 Groovy Voice Options for Kate
"""

import asyncio
import logging
from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class SimpleVoiceWidget(QWidget):
    """
    Simple voice widget with 3 groovy preset options instead of complex settings.
    """
    
    # Signals
    voice_test_requested = Signal(str, str)  # (text, voice_style)
    
    def __init__(self):
        super().__init__()
        self.current_voice = "natural"
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the simple voice UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("🎙️ Voice Assistant")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Voice Style Selection
        style_group = QGroupBox("Choose Your Voice Vibe")
        style_layout = QVBoxLayout(style_group)
        
        # Voice option buttons
        self.voice_buttons = {}
        
        # Option 1: Natural & Friendly
        natural_btn = self._create_voice_button(
            "🌟 Natural & Friendly",
            "Warm, conversational tone perfect for daily chats",
            "natural"
        )
        style_layout.addWidget(natural_btn)
        
        # Option 2: Professional & Clear  
        pro_btn = self._create_voice_button(
            "💼 Professional & Clear", 
            "Crisp, authoritative voice for business discussions",
            "professional"
        )
        style_layout.addWidget(pro_btn)
        
        # Option 3: Energetic & Fun
        fun_btn = self._create_voice_button(
            "🚀 Energetic & Fun",
            "Upbeat, enthusiastic tone to keep you motivated",
            "energetic"
        )
        style_layout.addWidget(fun_btn)
        
        layout.addWidget(style_group)
        
        # Test Section
        test_group = QGroupBox("Try It Out")
        test_layout = QVBoxLayout(test_group)
        
        self.test_text = QTextEdit()
        self.test_text.setPlainText("Hey there! I'm Kate, your AI assistant. How can I help you today?")
        self.test_text.setMaximumHeight(80)
        test_layout.addWidget(self.test_text)
        
        test_btn = QPushButton("🔊 Test Current Voice")
        test_btn.setFont(QFont("Arial", 11, QFont.Bold))
        test_btn.clicked.connect(self._test_voice)
        test_layout.addWidget(test_btn)
        
        layout.addWidget(test_group)
        
        # Set default selection
        self.voice_buttons["natural"].setChecked(True)
        self._apply_styling()
        
    def _create_voice_button(self, title: str, description: str, voice_id: str) -> QPushButton:
        """Create a voice option button."""
        btn = QPushButton()
        btn.setCheckable(True)
        btn.setMinimumHeight(80)
        
        # Create button text
        btn_text = f"{title}\n{description}"
        btn.setText(btn_text)
        btn.clicked.connect(lambda: self._select_voice(voice_id))
        
        self.voice_buttons[voice_id] = btn
        return btn
        
    def _select_voice(self, voice_id: str):
        """Select a voice style."""
        self.current_voice = voice_id
        
        # Update button states
        for vid, btn in self.voice_buttons.items():
            btn.setChecked(vid == voice_id)
            
        logger.info(f"Voice style selected: {voice_id}")
        
    def _test_voice(self):
        """Test the current voice."""
        text = self.test_text.toPlainText().strip()
        if text:
            self.voice_test_requested.emit(text, self.current_voice)
            
    def _apply_styling(self):
        """Apply groovy styling."""
        self.setStyleSheet("""
            SimpleVoiceWidget {
                background-color: #2b2b2b;
                border-radius: 8px;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin: 8px 0px;
                padding-top: 12px;
                color: #ffffff;
                background-color: #353535;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                color: #00d4ff;
                font-size: 12px;
            }
            
            QPushButton {
                background-color: #404040;
                border: 2px solid #555555;
                border-radius: 6px;
                color: #ffffff;
                padding: 12px;
                text-align: left;
                font-size: 11px;
                min-height: 60px;
            }
            
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #00d4ff;
            }
            
            QPushButton:checked {
                background-color: #0078d4;
                border-color: #00d4ff;
                color: #ffffff;
            }
            
            QPushButton:checked:hover {
                background-color: #106ebe;
            }
            
            QTextEdit {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #ffffff;
                padding: 8px;
                font-size: 10px;
            }
            
            QLabel {
                color: #ffffff;
            }
        """)
        
    def get_voice_config(self) -> Dict[str, any]:
        """Get the current voice configuration."""
        configs = {
            "natural": {
                "engine": "pyttsx3",
                "rate": 180,
                "volume": 0.8,
                "pitch": 0
            },
            "professional": {
                "engine": "pyttsx3", 
                "rate": 160,
                "volume": 0.9,
                "pitch": -10
            },
            "energetic": {
                "engine": "pyttsx3",
                "rate": 220,
                "volume": 0.95,
                "pitch": 15
            }
        }
        return configs.get(self.current_voice, configs["natural"])