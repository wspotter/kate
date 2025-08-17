"""
Database layer for Kate LLM Client.

This package contains all database-related components including:
- SQLAlchemy 2.0 async models
- Database manager for connection handling
- Migration support
"""

from .manager import DatabaseManager
from .models import Base, Conversation, Message, Assistant, FileAttachment

__all__ = [
    "DatabaseManager",
    "Base",
    "Conversation", 
    "Message",
    "Assistant",
    "FileAttachment",
]