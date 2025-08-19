"""
Central Settings Window for Kate LLM Desktop Client
Organized tabbed interface replacing scattered widget settings
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class VoiceSettingsTab(QWidget):
    """Voice settings tab with all voice controls organized properly."""
    
    def __init__(self):
        super().__init__()
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Voice Controls Group
        controls_group = QGroupBox("Voice Controls")
        controls_layout = QFormLayout(controls_group)
        
        # Mic on/off toggle
        self.mic_enabled = QCheckBox("Enable Microphone")
        self.mic_enabled.setChecked(True)
        controls_layout.addRow("Microphone:", self.mic_enabled)
        
        # Sound on/off toggle  
        self.sound_enabled = QCheckBox("Enable Sound Output")
        self.sound_enabled.setChecked(True)
        controls_layout.addRow("Sound:", self.sound_enabled)
        
        # Voice recognition mode
        self.recognition_mode = QComboBox()
        self.recognition_mode.addItems([
            "Push to Talk",
            "Continuous Listening", 
            "Wake Word Detection",
            "Manual Activation"
        ])
        self.recognition_mode.setCurrentText("Wake Word Detection")
        controls_layout.addRow("Recognition Mode:", self.recognition_mode)
        
        # Wake word input
        self.wake_word = QLineEdit("Hey Kate")
        self.wake_word.setPlaceholderText("Enter wake word (e.g., 'Hey Kate')")
        controls_layout.addRow("Wake Word:", self.wake_word)
        
        layout.addWidget(controls_group)
        
        # Voice Engine Group
        engine_group = QGroupBox("Voice Engine Settings")
        engine_layout = QFormLayout(engine_group)
        
        # Voice engine selection
        self.voice_engine = QComboBox()
        self.voice_engine.addItems([
            "pyttsx3 (System TTS)",
            "edge-tts (Microsoft)",
            "gTTS (Google)", 
            "System Default"
        ])
        engine_layout.addRow("TTS Engine:", self.voice_engine)
        
        # Voice speed/rate
        self.voice_rate = QSlider(Qt.Horizontal)
        self.voice_rate.setRange(50, 300)
        self.voice_rate.setValue(180)
        self.voice_rate_label = QLabel("180 WPM")
        self.voice_rate.valueChanged.connect(
            lambda v: self.voice_rate_label.setText(f"{v} WPM")
        )
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(self.voice_rate)
        rate_layout.addWidget(self.voice_rate_label)
        engine_layout.addRow("Speech Rate:", rate_layout)
        
        # Volume control
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(85)
        self.volume_label = QLabel("85%")
        self.volume.valueChanged.connect(
            lambda v: self.volume_label.setText(f"{v}%")
        )
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(self.volume)
        volume_layout.addWidget(self.volume_label)
        engine_layout.addRow("Volume:", volume_layout)
        
        layout.addWidget(engine_group)
        
        # Audio Device Group
        device_group = QGroupBox("Audio Devices")
        device_layout = QFormLayout(device_group)
        
        # Input device
        self.input_device = QComboBox()
        self.input_device.addItem("System Default")
        device_layout.addRow("Microphone Device:", self.input_device)
        
        # Output device
        self.output_device = QComboBox() 
        self.output_device.addItem("System Default")
        device_layout.addRow("Speaker Device:", self.output_device)
        
        layout.addWidget(device_group)
        
        # Test Controls
        test_group = QGroupBox("Test Voice System")
        test_layout = QVBoxLayout(test_group)
        
        self.test_text = QTextEdit()
        self.test_text.setPlainText("Hello! I'm Kate, your AI assistant. How can I help you today?")
        self.test_text.setMaximumHeight(80)
        test_layout.addWidget(self.test_text)
        
        test_buttons = QHBoxLayout()
        self.test_tts_btn = QPushButton("🔊 Test Voice Output")
        self.test_stt_btn = QPushButton("🎤 Test Voice Input")
        test_buttons.addWidget(self.test_tts_btn)
        test_buttons.addWidget(self.test_stt_btn)
        test_layout.addLayout(test_buttons)
        
        layout.addWidget(test_group)
        
        # Stretch to prevent cramping
        layout.addStretch()

    def get_settings(self) -> Dict[str, Any]:
        """Get current voice settings."""
        return {
            'mic_enabled': self.mic_enabled.isChecked(),
            'sound_enabled': self.sound_enabled.isChecked(),
            'recognition_mode': self.recognition_mode.currentText(),
            'wake_word': self.wake_word.text(),
            'voice_engine': self.voice_engine.currentText(),
            'voice_rate': self.voice_rate.value(),
            'volume': self.volume.value(),
            'input_device': self.input_device.currentText(),
            'output_device': self.output_device.currentText(),
        }
        
    def load_settings(self, settings: Dict[str, Any]):
        """Load settings into the UI."""
        self.mic_enabled.setChecked(settings.get('mic_enabled', True))
        self.sound_enabled.setChecked(settings.get('sound_enabled', True))
        
        mode = settings.get('recognition_mode', 'Wake Word Detection')
        idx = self.recognition_mode.findText(mode)
        if idx >= 0:
            self.recognition_mode.setCurrentIndex(idx)
            
        self.wake_word.setText(settings.get('wake_word', 'Hey Kate'))
        self.voice_rate.setValue(settings.get('voice_rate', 180))
        self.volume.setValue(settings.get('volume', 85))


class AgentSettingsTab(QWidget):
    """Agent selection and configuration tab."""
    
    def __init__(self):
        super().__init__()
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Agent Selection Group
        selection_group = QGroupBox("Agent Selection")
        selection_layout = QFormLayout(selection_group)
        
        # Simple agent dropdown
        self.agent_selector = QComboBox()
        self.agent_selector.addItems([
            "Kate - General Assistant",
            "Code Expert - Programming Help", 
            "Research Assistant - Analysis & Reports",
            "Creative Writer - Content Generation",
            "Math Tutor - Problem Solving",
            "Language Teacher - Learning Support"
        ])
        self.agent_selector.currentTextChanged.connect(self._on_agent_changed)
        selection_layout.addRow("Active Agent:", self.agent_selector)
        
        layout.addWidget(selection_group)
        
        # Agent Description Area
        desc_group = QGroupBox("Agent Description")
        desc_layout = QVBoxLayout(desc_group)
        
        self.agent_description = QTextEdit()
        self.agent_description.setReadOnly(True)
        self.agent_description.setMaximumHeight(120)
        self._update_agent_description()
        desc_layout.addWidget(self.agent_description)
        
        layout.addWidget(desc_group)
        
        # Agent Configuration (if needed)
        config_group = QGroupBox("Agent Configuration")
        config_layout = QFormLayout(config_group)
        
        self.agent_personality = QComboBox()
        self.agent_personality.addItems(["Professional", "Friendly", "Casual", "Expert"])
        config_layout.addRow("Personality:", self.agent_personality)
        
        self.response_length = QComboBox()
        self.response_length.addItems(["Concise", "Detailed", "Comprehensive"])
        config_layout.addRow("Response Style:", self.response_length)
        
        layout.addWidget(config_group)
        layout.addStretch()
        
    def _on_agent_changed(self):
        """Update description when agent changes."""
        self._update_agent_description()
        
    def _update_agent_description(self):
        """Update the agent description based on selection."""
        agent = self.agent_selector.currentText()
        descriptions = {
            "Kate - General Assistant": "Your versatile AI companion for everyday tasks, questions, and conversations. Kate can help with a wide range of topics and adapt to your needs.",
            "Code Expert - Programming Help": "Specialized in software development, debugging, code review, and technical problem-solving across multiple programming languages and frameworks.",
            "Research Assistant - Analysis & Reports": "Expert at gathering information, analyzing data, creating reports, and providing detailed research on any topic.",
            "Creative Writer - Content Generation": "Focused on creative writing, storytelling, content creation, and helping with all forms of written expression.",
            "Math Tutor - Problem Solving": "Specialized in mathematics, statistics, problem-solving, and explaining complex mathematical concepts clearly.",
            "Language Teacher - Learning Support": "Dedicated to language learning, grammar, vocabulary, pronunciation, and cultural understanding across multiple languages."
        }
        
        description = descriptions.get(agent, "Select an agent to see its description.")
        self.agent_description.setPlainText(description)

    def get_settings(self) -> Dict[str, Any]:
        """Get current agent settings."""
        return {
            'active_agent': self.agent_selector.currentText(),
            'agent_personality': self.agent_personality.currentText(),
            'response_length': self.response_length.currentText(),
        }


class AppSettingsTab(QWidget):
    """Application-level settings tab."""
    
    def __init__(self):
        super().__init__()
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Appearance Group
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout(appearance_group)
        
        self.theme = QComboBox()
        self.theme.addItems(["System Default", "Light", "Dark", "Auto"])
        appearance_layout.addRow("Theme:", self.theme)
        
        self.window_behavior = QComboBox()
        self.window_behavior.addItems(["Normal", "Always on Top", "Minimize to Tray"])
        appearance_layout.addRow("Window Behavior:", self.window_behavior)
        
        layout.addWidget(appearance_group)
        
        # Performance Group  
        performance_group = QGroupBox("Performance")
        performance_layout = QFormLayout(performance_group)
        
        self.auto_save = QCheckBox("Auto-save conversations")
        self.auto_save.setChecked(True)
        performance_layout.addRow("Auto-save:", self.auto_save)
        
        self.max_history = QSpinBox()
        self.max_history.setRange(10, 1000)
        self.max_history.setValue(100)
        self.max_history.setSuffix(" messages")
        performance_layout.addRow("Max History:", self.max_history)
        
        layout.addWidget(performance_group)
        layout.addStretch()

    def get_settings(self) -> Dict[str, Any]:
        """Get current app settings."""
        return {
            'theme': self.theme.currentText(),
            'window_behavior': self.window_behavior.currentText(),
            'auto_save': self.auto_save.isChecked(),
            'max_history': self.max_history.value(),
        }


class SettingsWindow(QDialog):
    """Central settings window with organized tabs."""
    
    settings_changed = Signal(dict)  # Emitted when settings change
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kate Settings")
        self.setModal(True)
        self.resize(600, 500)
        self._init_ui()
        self._load_settings()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tab widget with proper scrolling
        self.tab_widget = QTabWidget()
        
        # Voice settings tab
        self.voice_tab = VoiceSettingsTab()
        voice_scroll = QScrollArea()
        voice_scroll.setWidget(self.voice_tab)
        voice_scroll.setWidgetResizable(True)
        voice_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tab_widget.addTab(voice_scroll, "🎤 Voice")
        
        # Agent settings tab
        self.agent_tab = AgentSettingsTab()
        agent_scroll = QScrollArea()
        agent_scroll.setWidget(self.agent_tab)
        agent_scroll.setWidgetResizable(True)
        agent_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tab_widget.addTab(agent_scroll, "🤖 Agents")
        
        # App settings tab
        self.app_tab = AppSettingsTab()
        app_scroll = QScrollArea()
        app_scroll.setWidget(self.app_tab)
        app_scroll.setWidgetResizable(True)
        app_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tab_widget.addTab(app_scroll, "⚙️ Application")
        
        layout.addWidget(self.tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_settings)
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self._ok_clicked)
        self.ok_btn.setDefault(True)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
    def _load_settings(self):
        """Load settings from file."""
        settings_file = Path.home() / ".config" / "kate" / "settings.json"
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    
                # Load into each tab
                self.voice_tab.load_settings(settings.get('voice', {}))
                # Agent and app tabs would load their settings here
                
            except Exception as e:
                print(f"Error loading settings: {e}")
                
    def _save_settings(self):
        """Save settings to file."""
        settings = {
            'voice': self.voice_tab.get_settings(),
            'agent': self.agent_tab.get_settings(),
            'app': self.app_tab.get_settings(),
        }
        
        settings_file = Path.home() / ".config" / "kate" / "settings.json"
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
            
    def _apply_settings(self):
        """Apply current settings."""
        settings = {
            'voice': self.voice_tab.get_settings(),
            'agent': self.agent_tab.get_settings(),
            'app': self.app_tab.get_settings(),
        }
        
        self._save_settings()
        self.settings_changed.emit(settings)
        
    def _ok_clicked(self):
        """Handle OK button click."""
        self._apply_settings()
        self.accept()


# Standalone test function
def main():
    """Test the settings window."""
    app = QApplication([])
    
    window = SettingsWindow()
    window.show()
    
    app.exec()


if __name__ == "__main__":
    main()