"""
Base provider interface for Kate LLM Client.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel


class ProviderStatus(Enum):
    """Provider connection status."""
    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about an available model."""
    id: str
    name: str
    description: Optional[str] = None
    context_length: Optional[int] = None
    supports_streaming: bool = True
    supports_tools: bool = False
    parameters: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    """A chat message in the conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""
    messages: List[ChatMessage]
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Response from chat completion."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk from chat completion."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class defines the interface that all providers must implement
    to work with the Kate LLM Client.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.status = ProviderStatus.UNKNOWN
        self._models: List[ModelInfo] = []
        
    @property
    def is_connected(self) -> bool:
        """Check if the provider is connected and ready."""
        return self.status == ProviderStatus.CONNECTED
        
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the provider and validate credentials.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the provider and cleanup resources."""
        pass
        
    @abstractmethod
    async def get_models(self) -> List[ModelInfo]:
        """
        Get list of available models from the provider.
        
        Returns:
            List of available models
        """
        pass
        
    @abstractmethod
    async def chat_completion(
        self, 
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Generate a chat completion response.
        
        Args:
            request: Chat completion request parameters
            
        Returns:
            Chat completion response
        """
        pass
        
    @abstractmethod
    async def chat_completion_stream(
        self, 
        request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Generate a streaming chat completion response.
        
        Args:
            request: Chat completion request parameters
            
        Yields:
            Chat completion chunks
        """
        pass
        
    async def health_check(self) -> bool:
        """
        Perform a health check on the provider.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            models = await self.get_models()
            return len(models) > 0
        except Exception:
            return False
            
    async def validate_model(self, model_id: str) -> bool:
        """
        Validate that a model is available on this provider.
        
        Args:
            model_id: Model identifier to validate
            
        Returns:
            True if model is available, False otherwise
        """
        models = await self.get_models()
        return any(model.id == model_id for model in models)
        
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
        
    def update_status(self, status: ProviderStatus) -> None:
        """Update the provider status."""
        self.status = status