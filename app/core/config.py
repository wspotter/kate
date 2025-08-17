"""
Configuration management for Kate LLM Client.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import lru_cache

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import platformdirs


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


class AppSettings(BaseSettings):
    """Main application settings."""
    
    version: str = "1.0.0"
    debug: bool = False
    auto_save_interval: int = 300
    
    config_dir: Path = Field(default_factory=lambda: Path(platformdirs.user_config_dir("Kate")))
    data_dir: Path = Field(default_factory=lambda: Path(platformdirs.user_data_dir("Kate")))
    
    window: WindowSettings = Field(default_factory=WindowSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ui: UISettings = Field(default_factory=UISettings)
    
    class Config:
        env_prefix = "KATE_"
        env_nested_delimiter = "__"
        
    @validator("config_dir", "data_dir", pre=True)
    def ensure_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v
        
    @validator("database")
    def setup_database_path(cls, v, values):
        if v.url == "sqlite+aiosqlite:///kate.db":
            data_dir = values.get("data_dir", Path(platformdirs.user_data_dir("Kate")))
            db_path = data_dir / "kate.db"
            v.url = f"sqlite+aiosqlite:///{db_path}"
        return v
        
    def model_post_init(self, __context):
        for dir_path in [self.config_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> AppSettings:
    return AppSettings()