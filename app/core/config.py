"""Configuration management for Kate LLM Client."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import platformdirs
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class WindowSettings(BaseModel):
    """Window geometry and state settings."""
    width: int = 1200
    height: int = 800
    x: Optional[int] = None
    y: Optional[int] = None
    maximized: bool = False


class DatabaseSettings(BaseModel):
    """Database configuration settings."""
    url: str = "sqlite+aiosqlite:///kate.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


class UISettings(BaseModel):
    """UI appearance and behavior settings."""
    theme: str = "Kate Dark"
    font_family: str = "Segoe UI"
    font_size: int = 10
    sidebar_width: int = 300
    assistant_panel_width: int = 300


class VoiceSettings(BaseModel):
    """Voice and audio configuration settings."""
    
    # TTS (Text-to-Speech) Settings
    tts_enabled: bool = True
    tts_engine: str = "pyttsx3"  # "pyttsx3", "edge-tts", "gtts"
    tts_voice_id: Optional[str] = None  # Specific voice ID for selected engine
    tts_rate: int = 200  # Speech rate (words per minute)
    tts_volume: float = 0.9  # Volume level (0.0 to 1.0)
    tts_pitch: int = 0  # Pitch adjustment (-50 to +50)
    
    # Voice Choices for different engines
    pyttsx3_voice: str = "default"  # System default voice
    edge_tts_voice: str = "en-US-AriaNeural"  # Microsoft Edge TTS voice
    gtts_language: str = "en"  # Google TTS language code
    
    # STT (Speech-to-Text) Settings
    stt_enabled: bool = True
    stt_engine: str = "google"  # "google", "whisper", "sphinx"
    stt_language: str = "en-US"  # Language for speech recognition
    stt_timeout: float = 5.0  # Recording timeout in seconds
    stt_phrase_timeout: float = 0.3  # Pause detection timeout
    stt_energy_threshold: int = 300  # Microphone sensitivity
    stt_dynamic_energy_threshold: bool = True  # Auto-adjust microphone sensitivity
    
    # Audio Device Settings
    audio_input_device: Optional[str] = None  # Microphone device name
    audio_output_device: Optional[str] = None  # Speaker device name
    audio_sample_rate: int = 16000  # Sample rate for audio processing
    audio_chunk_size: int = 1024  # Audio buffer size
    
    # Advanced Settings
    enable_noise_reduction: bool = True  # Apply noise filtering
    enable_echo_cancellation: bool = True  # Cancel echo/feedback
    auto_adjust_microphone: bool = True  # Auto-adjust microphone levels
    voice_activation_threshold: float = 0.3  # Voice activation sensitivity
    
    # UI Integration Settings
    show_voice_controls: bool = True  # Show voice controls in UI
    show_audio_visualization: bool = True  # Show audio waveforms
    enable_push_to_talk: bool = False  # Require button press to talk
    push_to_talk_key: str = "space"  # Hotkey for push-to-talk


class AppSettings(BaseSettings):
    """Main application settings."""
    
    version: str = "1.0.0"
    debug: bool = False
    auto_save_interval: int = 300

    # --- Legacy flat fields (backward compatibility with older tests) ---
    # These mirror older configuration parameters expected directly on AppSettings.
    # They do not drive the new nested settings directly but can be referenced by
    # legacy components and tests. In the future, migrate tests to use nested models.
    database_url: str = "sqlite+aiosqlite:///kate.db"
    vector_store_path: str = "vector_store"
    document_store_path: str = "documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_retrieval_docs: int = 5
    similarity_threshold: float = 0.7
    
    config_dir: Path = Field(default_factory=lambda: Path(platformdirs.user_config_dir("Kate")))
    data_dir: Path = Field(default_factory=lambda: Path(platformdirs.user_data_dir("Kate")))
    
    window: WindowSettings = Field(default_factory=WindowSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ui: UISettings = Field(default_factory=UISettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    
    model_config = {
        "env_prefix": "KATE_",
        "env_nested_delimiter": "__",
        "extra": "allow",  # accept legacy unknown fields
    }
        
    @field_validator("config_dir", "data_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):  # type: ignore[override]
        return Path(v) if isinstance(v, str) else v

    @field_validator("database")
    @classmethod
    def setup_database_path(cls, v, values):  # type: ignore[override]
        try:
            if v.url == "sqlite+aiosqlite:///kate.db":
                data_dir = values.get("data_dir", Path(platformdirs.user_data_dir("Kate")))
                db_path = data_dir / "kate.db"
                v.url = f"sqlite+aiosqlite:///{db_path}"
        except Exception:
            pass
        return v
        
    def model_post_init(self, __context):
        for dir_path in [self.config_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        # If legacy database_url provided and default database.url untouched, sync it
        try:
            if self.database and self.database.url.endswith("kate.db") and self.database_url:
                # Use legacy value for active database configuration
                self.database.url = self.database_url  # type: ignore[attr-defined]
        except Exception:
            pass


@lru_cache()
def get_settings() -> AppSettings:
    return AppSettings()