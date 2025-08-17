"""
Base theme data structures for Kate LLM Client.
"""

from dataclasses import dataclass
from typing import List, Union


@dataclass
class ThemeData:
    """
    Complete theme data structure defining all visual aspects.
    """
    name: str
    display_name: str
    description: str
    
    # Color scheme
    primary: str
    secondary: str
    background: str
    surface: str
    error: str
    warning: str
    info: str
    success: str
    
    # Text colors
    text_primary: str
    text_secondary: str
    text_disabled: str
    text_hint: str
    
    # Interactive colors
    accent: str
    focus: str
    hover: str
    pressed: str
    selected: str
    disabled: str
    
    # Border and shadow
    border: str
    shadow: str
    
    # Typography
    font_family: str
    font_size: int
    font_weight: str
    line_height: float
    
    # Layout
    border_radius: int
    spacing: int
    padding: int
    margin: int
    container_padding: int
    
    # Component-specific colors
    sidebar_background: str
    chat_background: str
    message_user_background: str
    message_assistant_background: str
    input_background: str
    button_background: str
    
    def to_dict(self) -> dict:
        """Convert theme data to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ThemeData":
        """Create theme data from dictionary."""
        return cls(**data)


def get_kate_dark_theme() -> ThemeData:
    """Get the default Kate Dark theme."""
    return ThemeData(
        name="kate_dark",
        display_name="Kate Dark",
        description="Default dark theme for Kate LLM Client",
        
        # Color scheme
        primary="#2563eb",
        secondary="#64748b",
        background="#0f172a",
        surface="#1e293b",
        error="#ef4444",
        warning="#f59e0b",
        info="#3b82f6",
        success="#10b981",
        
        # Text colors
        text_primary="#f8fafc",
        text_secondary="#cbd5e1",
        text_disabled="#64748b",
        text_hint="#94a3b8",
        
        # Interactive colors
        accent="#3b82f6",
        focus="#60a5fa",
        hover="#334155",
        pressed="#1e293b",
        selected="#475569",
        disabled="#374151",
        
        # Border and shadow
        border="#374151",
        shadow="rgba(0, 0, 0, 0.5)",
        
        # Typography
        font_family="Segoe UI, system-ui, sans-serif",
        font_size=13,
        font_weight="400",
        line_height=1.4,
        
        # Layout
        border_radius=8,
        spacing=8,
        padding=12,
        margin=8,
        container_padding=16,
        
        # Component-specific colors
        sidebar_background="#1e293b",
        chat_background="#0f172a",
        message_user_background="#2563eb",
        message_assistant_background="#374151",
        input_background="#334155",
        button_background="#475569",
    )


def get_kate_light_theme() -> ThemeData:
    """Get the Kate Light theme."""
    return ThemeData(
        name="kate_light",
        display_name="Kate Light",
        description="Light theme for Kate LLM Client",
        
        # Color scheme
        primary="#2563eb",
        secondary="#64748b",
        background="#ffffff",
        surface="#f8fafc",
        error="#ef4444",
        warning="#f59e0b",
        info="#3b82f6",
        success="#10b981",
        
        # Text colors
        text_primary="#1e293b",
        text_secondary="#475569",
        text_disabled="#94a3b8",
        text_hint="#64748b",
        
        # Interactive colors
        accent="#3b82f6",
        focus="#60a5fa",
        hover="#f1f5f9",
        pressed="#e2e8f0",
        selected="#cbd5e1",
        disabled="#f1f5f9",
        
        # Border and shadow
        border="#e2e8f0",
        shadow="rgba(0, 0, 0, 0.1)",
        
        # Typography
        font_family="Segoe UI, system-ui, sans-serif",
        font_size=13,
        font_weight="400",
        line_height=1.4,
        
        # Layout
        border_radius=8,
        spacing=8,
        padding=12,
        margin=8,
        container_padding=16,
        
        # Component-specific colors
        sidebar_background="#f8fafc",
        chat_background="#ffffff",
        message_user_background="#2563eb",
        message_assistant_background="#f1f5f9",
        input_background="#ffffff",
        button_background="#f8fafc",
    )