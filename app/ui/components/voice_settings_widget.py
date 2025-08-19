"""
Voice Settings Widget for Kate LLM Client.
Provides comprehensive voice configuration options.
"""

import logging
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class VoiceSettingsWidget(QWidget):
    """
    Comprehensive voice settings configuration widget.
    """
    
    # Signals
    settings_changed = Signal(dict)  # Emitted when any setting changes
    test_tts_requested = Signal(str)  # Emitted when TTS test is requested
    test_stt_requested = Signal()    # Emitted when STT test is requested
    
    def __init__(self, voice_settings=None):
        super().__init__()
        self.voice_settings = voice_settings
        self.logger = logging.getLogger(__name__)
        
        # Available voice options (populated dynamically)
        self.available_tts_engines = ["pyttsx3", "edge-tts", "gtts"]
        self.available_stt_engines = ["google", "whisper", "sphinx"]
        self.available_voices = {
            "pyttsx3": ["default", "male", "female"],
            "edge-tts": [
                "en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural",
                "en-US-AndrewNeural", "en-US-EmmaNeural", "en-US-BrianNeural",
                "en-GB-LibbyNeural", "en-GB-MaisieNeural", "en-GB-RyanNeural",
                "en-AU-NatashaNeural", "en-AU-WilliamNeural", "en-CA-ClaraNeural"
            ],
            "gtts": ["en", "en-us", "en-uk", "en-au", "en-ca", "en-in"]
        }
        
        self._setup_ui()
        self._setup_connections()
        self._load_settings()
        
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget for different categories
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # TTS Settings Tab
        self.tts_tab = self._create_tts_settings()
        self.tabs.addTab(self.tts_tab, "🔊 Text-to-Speech")
        
        # STT Settings Tab
        self.stt_tab = self._create_stt_settings()
        self.tabs.addTab(self.stt_tab, "🎙️ Speech-to-Text")
        
        # Audio Devices Tab
        self.audio_tab = self._create_audio_settings()
        self.tabs.addTab(self.audio_tab, "🔧 Audio Devices")
        
        # Advanced Settings Tab
        self.advanced_tab = self._create_advanced_settings()
        self.tabs.addTab(self.advanced_tab, "⚙️ Advanced")
        
        # Test Panel
        self.test_panel = self._create_test_panel()
        layout.addWidget(self.test_panel)
        
    def _create_tts_settings(self) -> QWidget:
        """Create TTS settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # TTS Enable/Disable
        self.tts_enabled = QCheckBox("Enable Text-to-Speech")
        self.tts_enabled.setChecked(True)
        layout.addWidget(self.tts_enabled)
        
        # Engine Selection
        engine_group = QGroupBox("TTS Engine")
        engine_layout = QFormLayout(engine_group)
        
        self.tts_engine = QComboBox()
        self.tts_engine.addItems(self.available_tts_engines)
        engine_layout.addRow("Engine:", self.tts_engine)
        
        layout.addWidget(engine_group)
        
        # Voice Selection
        voice_group = QGroupBox("Voice Selection")
        voice_layout = QFormLayout(voice_group)
        
        self.voice_combo = QComboBox()
        voice_layout.addRow("Voice:", self.voice_combo)
        
        layout.addWidget(voice_group)
        
        # Voice Parameters
        params_group = QGroupBox("Voice Parameters")
        params_layout = QFormLayout(params_group)
        
        # Speech Rate
        self.tts_rate = QSpinBox()
        self.tts_rate.setRange(50, 400)
        self.tts_rate.setValue(200)
        self.tts_rate.setSuffix(" wpm")
        params_layout.addRow("Speech Rate:", self.tts_rate)
        
        # Volume
        self.tts_volume = QSlider(Qt.Horizontal)
        self.tts_volume.setRange(0, 100)
        self.tts_volume.setValue(90)
        self.volume_label = QLabel("90%")
        
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(self.tts_volume)
        volume_layout.addWidget(self.volume_label)
        params_layout.addRow("Volume:", volume_layout)
        
        # Pitch
        self.tts_pitch = QSlider(Qt.Horizontal)
        self.tts_pitch.setRange(-50, 50)
        self.tts_pitch.setValue(0)
        self.pitch_label = QLabel("0")
        
        pitch_layout = QHBoxLayout()
        pitch_layout.addWidget(self.tts_pitch)
        pitch_layout.addWidget(self.pitch_label)
        params_layout.addRow("Pitch:", pitch_layout)
        
        layout.addWidget(params_group)
        
        # Add stretch at the end
        layout.addStretch()
        
        return widget
        
    def _create_stt_settings(self) -> QWidget:
        """Create STT settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # STT Enable/Disable
        self.stt_enabled = QCheckBox("Enable Speech-to-Text")
        self.stt_enabled.setChecked(True)
        layout.addWidget(self.stt_enabled)
        
        # Engine Selection
        engine_group = QGroupBox("STT Engine")
        engine_layout = QFormLayout(engine_group)
        
        self.stt_engine = QComboBox()
        self.stt_engine.addItems(self.available_stt_engines)
        engine_layout.addRow("Engine:", self.stt_engine)
        
        # Language
        self.stt_language = QComboBox()
        self.stt_language.addItems([
            "en-US", "en-GB", "en-AU", "en-CA", "en-IN",
            "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR",
            "ja-JP", "ko-KR", "zh-CN", "ru-RU", "ar-SA"
        ])
        engine_layout.addRow("Language:", self.stt_language)
        
        layout.addWidget(engine_group)
        
        # Recognition Parameters
        params_group = QGroupBox("Recognition Parameters")
        params_layout = QFormLayout(params_group)
        
        # Timeout
        self.stt_timeout = QDoubleSpinBox()
        self.stt_timeout.setRange(1.0, 30.0)
        self.stt_timeout.setValue(5.0)
        self.stt_timeout.setSuffix(" seconds")
        params_layout.addRow("Recording Timeout:", self.stt_timeout)
        
        # Phrase Timeout
        self.stt_phrase_timeout = QDoubleSpinBox()
        self.stt_phrase_timeout.setRange(0.1, 2.0)
        self.stt_phrase_timeout.setValue(0.3)
        self.stt_phrase_timeout.setSuffix(" seconds")
        params_layout.addRow("Phrase Timeout:", self.stt_phrase_timeout)
        
        # Energy Threshold
        self.stt_energy = QSpinBox()
        self.stt_energy.setRange(100, 2000)
        self.stt_energy.setValue(300)
        params_layout.addRow("Energy Threshold:", self.stt_energy)
        
        # Dynamic Energy
        self.stt_dynamic_energy = QCheckBox("Auto-adjust sensitivity")
        self.stt_dynamic_energy.setChecked(True)
        params_layout.addRow("", self.stt_dynamic_energy)
        
        layout.addWidget(params_group)
        layout.addStretch()
        
        return widget
        
    def _create_audio_settings(self) -> QWidget:
        """Create audio device settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Input Device
        input_group = QGroupBox("Input Device (Microphone)")
        input_layout = QFormLayout(input_group)
        
        self.input_device = QComboBox()
        self.input_device.addItems(["Default", "System Microphone", "USB Microphone", "Bluetooth Headset"])
        input_layout.addRow("Input Device:", self.input_device)
        
        layout.addWidget(input_group)
        
        # Output Device
        output_group = QGroupBox("Output Device (Speakers)")
        output_layout = QFormLayout(output_group)
        
        self.output_device = QComboBox()
        self.output_device.addItems(["Default", "System Speakers", "Headphones", "Bluetooth Speaker"])
        output_layout.addRow("Output Device:", self.output_device)
        
        layout.addWidget(output_group)
        
        # Audio Quality
        quality_group = QGroupBox("Audio Quality")
        quality_layout = QFormLayout(quality_group)
        
        self.sample_rate = QComboBox()
        self.sample_rate.addItems(["8000 Hz", "16000 Hz", "22050 Hz", "44100 Hz", "48000 Hz"])
        self.sample_rate.setCurrentText("16000 Hz")
        quality_layout.addRow("Sample Rate:", self.sample_rate)
        
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(256, 4096)
        self.chunk_size.setValue(1024)
        quality_layout.addRow("Buffer Size:", self.chunk_size)
        
        layout.addWidget(quality_group)
        layout.addStretch()
        
        return widget
        
    def _create_advanced_settings(self) -> QWidget:
        """Create advanced settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Audio Processing
        processing_group = QGroupBox("Audio Processing")
        processing_layout = QVBoxLayout(processing_group)
        
        self.noise_reduction = QCheckBox("Enable noise reduction")
        self.noise_reduction.setChecked(True)
        processing_layout.addWidget(self.noise_reduction)
        
        self.echo_cancellation = QCheckBox("Enable echo cancellation")
        self.echo_cancellation.setChecked(True)
        processing_layout.addWidget(self.echo_cancellation)
        
        self.auto_adjust_mic = QCheckBox("Auto-adjust microphone levels")
        self.auto_adjust_mic.setChecked(True)
        processing_layout.addWidget(self.auto_adjust_mic)
        
        layout.addWidget(processing_group)
        
        # Voice Activation
        activation_group = QGroupBox("Voice Activation")
        activation_layout = QFormLayout(activation_group)
        
        self.voice_threshold = QSlider(Qt.Horizontal)
        self.voice_threshold.setRange(10, 90)
        self.voice_threshold.setValue(30)
        self.threshold_label = QLabel("30%")
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.voice_threshold)
        threshold_layout.addWidget(self.threshold_label)
        activation_layout.addRow("Activation Threshold:", threshold_layout)
        
        layout.addWidget(activation_group)
        
        # UI Integration
        ui_group = QGroupBox("UI Integration")
        ui_layout = QVBoxLayout(ui_group)
        
        self.show_voice_controls = QCheckBox("Show voice controls in main UI")
        self.show_voice_controls.setChecked(True)
        ui_layout.addWidget(self.show_voice_controls)
        
        self.show_audio_visualization = QCheckBox("Show audio waveforms")
        self.show_audio_visualization.setChecked(True)
        ui_layout.addWidget(self.show_audio_visualization)
        
        self.push_to_talk = QCheckBox("Enable push-to-talk mode")
        self.push_to_talk.setChecked(False)
        ui_layout.addWidget(self.push_to_talk)
        
        layout.addWidget(ui_group)
        layout.addStretch()
        
        return widget
        
    def _create_test_panel(self) -> QWidget:
        """Create voice testing panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Test TTS
        tts_group = QGroupBox("Test Text-to-Speech")
        tts_layout = QVBoxLayout(tts_group)
        
        self.test_text = QTextEdit()
        self.test_text.setPlainText("Hello! This is a test of the text-to-speech system. How do I sound?")
        self.test_text.setMaximumHeight(80)
        tts_layout.addWidget(self.test_text)
        
        self.test_tts_btn = QPushButton("🔊 Test TTS")
        self.test_tts_btn.setMaximumWidth(120)
        tts_layout.addWidget(self.test_tts_btn)
        
        layout.addWidget(tts_group)
        
        # Test STT
        stt_group = QGroupBox("Test Speech-to-Text")
        stt_layout = QVBoxLayout(stt_group)
        
        self.test_stt_btn = QPushButton("🎙️ Test STT (5 seconds)")
        self.test_stt_btn.setMaximumWidth(180)
        stt_layout.addWidget(self.test_stt_btn)
        
        self.stt_result = QTextEdit()
        self.stt_result.setReadOnly(True)
        self.stt_result.setMaximumHeight(60)
        self.stt_result.setPlaceholderText("STT results will appear here...")
        stt_layout.addWidget(self.stt_result)
        
        layout.addWidget(stt_group)
        
        return panel
        
    def _setup_connections(self):
        """Setup signal connections."""
        # TTS connections
        self.tts_engine.currentTextChanged.connect(self._on_tts_engine_changed)
        self.tts_enabled.toggled.connect(self._emit_settings_changed)
        self.tts_rate.valueChanged.connect(self._emit_settings_changed)
        self.tts_volume.valueChanged.connect(self._on_volume_changed)
        self.tts_pitch.valueChanged.connect(self._on_pitch_changed)
        
        # STT connections
        self.stt_enabled.toggled.connect(self._emit_settings_changed)
        self.stt_engine.currentTextChanged.connect(self._emit_settings_changed)
        self.stt_language.currentTextChanged.connect(self._emit_settings_changed)
        self.stt_timeout.valueChanged.connect(self._emit_settings_changed)
        
        # Advanced connections
        self.voice_threshold.valueChanged.connect(self._on_threshold_changed)
        
        # Test connections
        self.test_tts_btn.clicked.connect(self._test_tts)
        self.test_stt_btn.clicked.connect(self._test_stt)
        
    def _on_tts_engine_changed(self, engine: str):
        """Handle TTS engine change."""
        self.voice_combo.clear()
        if engine in self.available_voices:
            self.voice_combo.addItems(self.available_voices[engine])
        self._emit_settings_changed()
        
    def _on_volume_changed(self, value: int):
        """Handle volume slider change."""
        self.volume_label.setText(f"{value}%")
        self._emit_settings_changed()
        
    def _on_pitch_changed(self, value: int):
        """Handle pitch slider change."""
        self.pitch_label.setText(str(value))
        self._emit_settings_changed()
        
    def _on_threshold_changed(self, value: int):
        """Handle threshold slider change."""
        self.threshold_label.setText(f"{value}%")
        self._emit_settings_changed()
        
    def _emit_settings_changed(self):
        """Emit settings changed signal with current values."""
        settings = self.get_current_settings()
        self.settings_changed.emit(settings)
        
    def _test_tts(self):
        """Test TTS with current settings."""
        text = self.test_text.toPlainText().strip()
        if text:
            self.test_tts_requested.emit(text)
            
    def _test_stt(self):
        """Test STT with current settings."""
        self.test_stt_requested.emit()
        
    def get_current_settings(self) -> Dict:
        """Get current voice settings as dictionary."""
        return {
            # TTS Settings
            "tts_enabled": self.tts_enabled.isChecked(),
            "tts_engine": self.tts_engine.currentText(),
            "tts_voice_id": self.voice_combo.currentText(),
            "tts_rate": self.tts_rate.value(),
            "tts_volume": self.tts_volume.value() / 100.0,
            "tts_pitch": self.tts_pitch.value(),
            
            # STT Settings
            "stt_enabled": self.stt_enabled.isChecked(),
            "stt_engine": self.stt_engine.currentText(),
            "stt_language": self.stt_language.currentText(),
            "stt_timeout": self.stt_timeout.value(),
            "stt_phrase_timeout": self.stt_phrase_timeout.value(),
            "stt_energy_threshold": self.stt_energy.value(),
            "stt_dynamic_energy_threshold": self.stt_dynamic_energy.isChecked(),
            
            # Audio Settings
            "audio_input_device": self.input_device.currentText(),
            "audio_output_device": self.output_device.currentText(),
            "audio_sample_rate": int(self.sample_rate.currentText().split()[0]),
            "audio_chunk_size": self.chunk_size.value(),
            
            # Advanced Settings
            "enable_noise_reduction": self.noise_reduction.isChecked(),
            "enable_echo_cancellation": self.echo_cancellation.isChecked(),
            "auto_adjust_microphone": self.auto_adjust_mic.isChecked(),
            "voice_activation_threshold": self.voice_threshold.value() / 100.0,
            "show_voice_controls": self.show_voice_controls.isChecked(),
            "show_audio_visualization": self.show_audio_visualization.isChecked(),
            "enable_push_to_talk": self.push_to_talk.isChecked(),
        }
        
    def _load_settings(self):
        """Load settings from voice_settings object."""
        if not self.voice_settings:
            return
            
        try:
            # TTS Settings
            self.tts_enabled.setChecked(getattr(self.voice_settings, 'tts_enabled', True))
            
            engine = getattr(self.voice_settings, 'tts_engine', 'pyttsx3')
            if engine in self.available_tts_engines:
                self.tts_engine.setCurrentText(engine)
                
            self.tts_rate.setValue(getattr(self.voice_settings, 'tts_rate', 200))
            self.tts_volume.setValue(int(getattr(self.voice_settings, 'tts_volume', 0.9) * 100))
            self.tts_pitch.setValue(getattr(self.voice_settings, 'tts_pitch', 0))
            
            # STT Settings
            self.stt_enabled.setChecked(getattr(self.voice_settings, 'stt_enabled', True))
            
            stt_engine = getattr(self.voice_settings, 'stt_engine', 'google')
            if stt_engine in self.available_stt_engines:
                self.stt_engine.setCurrentText(stt_engine)
                
            self.stt_language.setCurrentText(getattr(self.voice_settings, 'stt_language', 'en-US'))
            self.stt_timeout.setValue(getattr(self.voice_settings, 'stt_timeout', 5.0))
            
            # Advanced settings
            self.noise_reduction.setChecked(getattr(self.voice_settings, 'enable_noise_reduction', True))
            self.show_voice_controls.setChecked(getattr(self.voice_settings, 'show_voice_controls', True))
            
        except Exception as e:
            self.logger.warning(f"Error loading voice settings: {e}")
            
    def update_stt_result(self, text: str):
        """Update the STT test result display."""
        self.stt_result.setPlainText(text)