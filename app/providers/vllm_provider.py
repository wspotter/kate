"""
vLLM provider for Kate LLM Client.

This provider connects to a local vLLM server running with OpenAI-compatible API.
vLLM provides high-performance inference for open-source models with features like:
- Continuous batching
- Optimized attention mechanisms  
- High throughput serving
- OpenAI API compatibility
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from urllib.parse import urljoin

import httpx
from loguru import logger

from .base import (
    BaseLLMProvider, 
    ModelInfo, 
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionChunk,
    ProviderStatus
)


class VLLMProvider(BaseLLMProvider):
    """
    vLLM provider for high-performance local model inference.
    
    Connects to a vLLM server running with OpenAI-compatible API endpoints.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("vLLM", config)
        
        # vLLM server configuration
        self.base_url = config.get("base_url", "http://localhost:8000")
        self.api_key = config.get("api_key", "EMPTY")  # vLLM uses "EMPTY" as default
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        
        # Logger
        self.logger = logger.bind(component="VLLMProvider")
        
    async def connect(self) -> bool:
        """
        Connect to the vLLM server and validate it's running.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.update_status(ProviderStatus.CONNECTING)
            self.logger.info(f"Connecting to vLLM server at {self.base_url}")
            
            # Create HTTP client
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            # Test connection by getting models
            await self.get_models()
            
            self.update_status(ProviderStatus.CONNECTED)
            self.logger.info("Successfully connected to vLLM server")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to vLLM server: {e}")
            self.update_status(ProviderStatus.ERROR)
            return False
            
    async def disconnect(self) -> None:
        """Disconnect from the vLLM server and cleanup resources."""
        try:
            if self.client:
                await self.client.aclose()
                self.client = None
                
            self.update_status(ProviderStatus.DISCONNECTED)
            self.logger.info("Disconnected from vLLM server")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
            
    async def get_models(self) -> List[ModelInfo]:
        """
        Get list of available models from the vLLM server.
        
        Returns:
            List of available models
        """
        if not self.client:
            raise RuntimeError("Not connected to vLLM server")
            
        try:
            response = await self.client.get("/v1/models")
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get("data", []):
                model_info = ModelInfo(
                    id=model_data["id"],
                    name=model_data.get("id", "Unknown"),
                    description=f"vLLM model: {model_data['id']}",
                    context_length=model_data.get("context_length"),
                    supports_streaming=True,
                    supports_tools=False,  # Most vLLM models don't support tools yet
                    parameters={
                        "created": model_data.get("created"),
                        "object": model_data.get("object"),
                        "owned_by": model_data.get("owned_by", "vllm")
                    }
                )
                models.append(model_info)
                
            self._models = models
            self.logger.debug(f"Found {len(models)} models on vLLM server")
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to get models from vLLM server: {e}")
            raise
            
    async def chat_completion(
        self, 
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Generate a chat completion response using vLLM.
        
        Args:
            request: Chat completion request parameters
            
        Returns:
            Chat completion response
        """
        if not self.client:
            raise RuntimeError("Not connected to vLLM server")
            
        try:
            # Convert request to vLLM format
            payload = {
                "model": request.model,
                "messages": [msg.dict() for msg in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stream": False
            }
            
            # Add optional parameters
            if request.stop:
                payload["stop"] = request.stop
                
            self.logger.debug(f"Sending chat completion request to vLLM: {request.model}")
            
            response = await self.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to our response format
            return ChatCompletionResponse(**data)
            
        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}")
            raise
            
    async def chat_completion_stream(
        self, 
        request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Generate a streaming chat completion response using vLLM.
        
        Args:
            request: Chat completion request parameters
            
        Yields:
            Chat completion chunks
        """
        if not self.client:
            raise RuntimeError("Not connected to vLLM server")
            
        try:
            # Convert request to vLLM format
            payload = {
                "model": request.model,
                "messages": [msg.dict() for msg in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stream": True
            }
            
            # Add optional parameters
            if request.stop:
                payload["stop"] = request.stop
                
            self.logger.debug(f"Sending streaming chat completion request to vLLM: {request.model}")
            
            async with self.client.stream(
                "POST", 
                "/v1/chat/completions", 
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str.strip() == "[DONE]":
                            break
                            
                        try:
                            data = json.loads(data_str)
                            chunk = ChatCompletionChunk(**data)
                            yield chunk
                            
                        except json.JSONDecodeError:
                            self.logger.warning(f"Failed to parse streaming response: {data_str}")
                            continue
                            
        except Exception as e:
            self.logger.error(f"Streaming chat completion failed: {e}")
            raise
            
    async def health_check(self) -> bool:
        """
        Perform a health check on the vLLM server.
        
        Returns:
            True if vLLM server is healthy, False otherwise
        """
        try:
            if not self.client:
                return False
                
            # Simple health check - try to get models
            response = await self.client.get("/v1/models", timeout=5)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
            
    async def get_server_info(self) -> Dict[str, Any]:
        """
        Get information about the vLLM server.
        
        Returns:
            Server information dictionary
        """
        if not self.client:
            raise RuntimeError("Not connected to vLLM server")
            
        try:
            # vLLM doesn't have a standard info endpoint, so we'll gather what we can
            models_response = await self.client.get("/v1/models")
            models_data = models_response.json()
            
            return {
                "provider": "vLLM",
                "base_url": self.base_url,
                "status": self.status.value,
                "models_count": len(models_data.get("data", [])),
                "available_models": [model["id"] for model in models_data.get("data", [])],
                "capabilities": {
                    "chat_completions": True,
                    "streaming": True,
                    "embeddings": False,  # Most vLLM setups don't support embeddings
                    "function_calling": False
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get server info: {e}")
            raise
            
    def get_recommended_settings(self, model_id: str) -> Dict[str, Any]:
        """
        Get recommended settings for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Recommended settings dictionary
        """
        # Basic recommendations for vLLM models
        base_settings = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        # Model-specific optimizations
        if "instruct" in model_id.lower() or "chat" in model_id.lower():
            base_settings.update({
                "temperature": 0.7,
                "top_p": 0.9
            })
        elif "code" in model_id.lower():
            base_settings.update({
                "temperature": 0.2,
                "top_p": 0.95
            })
            
        return base_settings