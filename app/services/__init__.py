"""
Services for Kate LLM Client.

This package contains business logic services including:
- Conversation management
- Search functionality
- Update management
- Assistant management
- Translation services
"""

from .search_service import SearchService
from .update_manager import UpdateManager

__all__ = [
    "SearchService",
    "UpdateManager",
]