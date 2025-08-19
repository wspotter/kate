"""
Event system for Kate LLM Client.

Provides a type-safe event bus for decoupled communication between components.
"""

import asyncio
from abc import ABC
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    cast,
)

from loguru import logger

# Event type variable for type safety
EventType = TypeVar('EventType', bound='Event')


@dataclass
class Event(ABC):
    """Base event class."""
    pass


# Application Events
@dataclass
class ApplicationStartedEvent(Event):
    """Emitted when the application starts."""
    pass


@dataclass
class ApplicationShutdownEvent(Event):
    """Emitted when the application is shutting down."""
    pass


# Conversation Events
@dataclass
class ConversationCreatedEvent(Event):
    """Emitted when a new conversation is created."""
    conversation_id: str
    title: str


@dataclass
class ConversationDeletedEvent(Event):
    """Emitted when a conversation is deleted."""
    conversation_id: str


@dataclass
class ConversationSelectedEvent(Event):
    """Emitted when a conversation is selected."""
    conversation_id: str


# Message Events
@dataclass
class MessageSentEvent(Event):
    """Emitted when a message is sent."""
    conversation_id: str
    message_id: str
    content: str


@dataclass
class MessageReceivedEvent(Event):
    """Emitted when a message is received from an LLM."""
    conversation_id: str
    message_id: str
    content: str
    provider: str
    model: str


@dataclass
class MessageStreamingEvent(Event):
    """Emitted during message streaming."""
    conversation_id: str
    message_id: str
    chunk: str
    is_complete: bool = False


# Provider Events
@dataclass
class ProviderConnectedEvent(Event):
    """Emitted when a provider connects."""
    provider_name: str
    models: List[str]


@dataclass
class ProviderDisconnectedEvent(Event):
    """Emitted when a provider disconnects."""
    provider_name: str
    reason: str


@dataclass
class ProviderErrorEvent(Event):
    """Emitted when a provider encounters an error."""
    provider_name: str
    error: str
    conversation_id: Optional[str] = None


# Theme Events
@dataclass
class ThemeChangedEvent(Event):
    """Emitted when the theme changes."""
    theme_name: str


# Update Events
@dataclass
class UpdateCheckStartedEvent(Event):
    """Emitted when an update check starts."""
    manual: bool = False


@dataclass
class UpdateAvailableEvent(Event):
    """Emitted when an update is available."""
    current_version: str
    latest_version: str
    download_url: str
    release_notes: str


@dataclass
class UpdateCompletedEvent(Event):
    """Emitted when an update completes."""
    success: bool
    error: Optional[str] = None


# Search Events
@dataclass
class SearchStartedEvent(Event):
    """Emitted when a search starts."""
    query: str
    search_type: str


@dataclass
class SearchCompletedEvent(Event):
    """Emitted when a search completes."""
    query: str
    results: List[Dict[str, Any]]
    search_type: str


class EventBus:
    """
    Event bus for type-safe event handling.
    
    Supports both synchronous and asynchronous event handlers.
    """

    _handlers: Dict[Type[Event], List[Callable[[Event], None]]]
    _async_handlers: Dict[Type[Event], List[Callable[[Event], Awaitable[Any]]]]
    logger: Any
    
    def __init__(self):
        self._handlers = {}
        self._async_handlers = {}
        self.logger = logger.bind(component="EventBus")
        
    def subscribe(self, event_type: Type[Event], handler: Callable[[Event], None]) -> None:
        """
        Subscribe to an event type with a synchronous handler.
        
        Args:
            event_type: The type of event to listen for
            handler: Function to call when event is emitted
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        self.logger.debug(f"Subscribed sync handler for {event_type.__name__}")
        
    def subscribe_async(self, event_type: Type[Event], handler: Callable[[Event], Awaitable[Any]]) -> None:
        """
        Subscribe to an event type with an asynchronous handler.
        
        Args:
            event_type: The type of event to listen for
            handler: Async function to call when event is emitted
        """
        if event_type not in self._async_handlers:
            self._async_handlers[event_type] = []
        self._async_handlers[event_type].append(handler)
        self.logger.debug(f"Subscribed async handler for {event_type.__name__}")
        
    def unsubscribe(self, event_type: Type[Event], handler: Callable[[Event], Any]) -> None:
        """
        Unsubscribe a handler from an event type.
        
        Args:
            event_type: The type of event to unsubscribe from
            handler: The handler function to remove
        """
        # Remove from sync handlers
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                self.logger.debug(f"Unsubscribed sync handler for {event_type.__name__}")
            except ValueError:
                pass
                
        # Remove from async handlers
        if event_type in self._async_handlers:
            try:
                self._async_handlers[event_type].remove(handler)
                self.logger.debug(f"Unsubscribed async handler for {event_type.__name__}")
            except ValueError:
                pass
                
    def emit(self, event: Event) -> None:
        """
        Emit an event synchronously.
        
        Args:
            event: The event to emit
        """
        event_type = type(event)
        self.logger.debug(f"Emitting {event_type.__name__}")
        
        # Call synchronous handlers
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Error in sync event handler for {event_type.__name__}: {e}")
                    
    async def emit_async(self, event: Event) -> None:
        """
        Emit an event asynchronously.
        
        Args:
            event: The event to emit
        """
        event_type = type(event)
        self.logger.debug(f"Emitting async {event_type.__name__}")
        
        # Call synchronous handlers first
        self.emit(event)
        
        # Call asynchronous handlers
        if event_type in self._async_handlers:
            tasks: List[asyncio.Task[Any]] = []
            for handler in self._async_handlers[event_type]:
                try:
                    coro = handler(event)
                    # Ensure we pass a proper coroutine to create_task
                    task = asyncio.create_task(cast(Coroutine[Any, Any, Any], coro))
                    tasks.append(task)
                except Exception as e:
                    self.logger.error(f"Error creating task for async event handler for {event_type.__name__}: {e}")
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                    
    def clear_handlers(self) -> None:
        """Clear all event handlers."""
        self._handlers.clear()
        self._async_handlers.clear()
        self.logger.debug("Cleared all event handlers")
        
    def get_handler_count(self, event_type: Type[Event]) -> int:
        """
        Get the number of handlers for an event type.
        
        Args:
            event_type: The event type to check
            
        Returns:
            Total number of handlers (sync + async)
        """
        sync_count = len(self._handlers.get(event_type, []))
        async_count = len(self._async_handlers.get(event_type, []))
        return sync_count + async_count
        
    def list_event_types(self) -> List[Type[Event]]:
        """
        Get a list of all event types that have handlers.
        
        Returns:
            List of event types
        """
        event_types: Set[Type[Event]] = set()
        event_types.update(self._handlers.keys())
        event_types.update(self._async_handlers.keys())
        return list(event_types)