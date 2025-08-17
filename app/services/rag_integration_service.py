"""
RAG integration service for Kate LLM Client.

Orchestrates the complete RAG workflow integrating retrieval, generation,
and UI components for seamless RAG-enhanced conversations.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from loguru import logger

from ..core.events import EventBus
from ..database.manager import DatabaseManager
from ..database.models import Conversation, Message
from ..providers.base import BaseLLMProvider
from ..providers.vllm_provider import VLLMProvider
from .conversation_embedding_service import ConversationEmbeddingService
from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .rag_config_service import RAGConfigService
from .rag_evaluation_service import RAGEvaluationService, ResponseEvaluation, RetrievalContext
from .rag_service import RAGMode, RAGRequest, RAGResponse, RAGService
from .retrieval_service import RetrievalService
from .vector_store import VectorStore


@dataclass
class ChatContext:
    """Context for a chat conversation."""
    conversation_id: str
    user_id: Optional[str] = None
    assistant_id: Optional[str] = None
    provider_id: Optional[str] = None
    model_name: Optional[str] = None
    rag_enabled: bool = True
    rag_mode: RAGMode = RAGMode.HYBRID


@dataclass
class ResponseMetrics:
    """Metrics for response generation."""
    retrieval_time_ms: int
    generation_time_ms: int
    total_time_ms: int
    sources_count: int
    context_relevance: float
    response_length: int


class RAGIntegrationService:
    """
    Service that integrates RAG capabilities with the chat system.
    
    Orchestrates the complete workflow from user message to RAG-enhanced response,
    managing all components and providing seamless integration with the UI.
    """
    
    def __init__(
        self,
        database_manager: DatabaseManager,
        event_bus: EventBus,
        rag_config_service: RAGConfigService,
        rag_service: Optional[RAGService] = None,
        retrieval_service: Optional[RetrievalService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        document_processor: Optional[DocumentProcessor] = None,
        conversation_embedding_service: Optional[ConversationEmbeddingService] = None,
        rag_evaluation_service: Optional[RAGEvaluationService] = None
    ):
        self.database_manager = database_manager
        self.event_bus = event_bus
        self.rag_config_service = rag_config_service
        self.rag_service = rag_service
        self.retrieval_service = retrieval_service
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.document_processor = document_processor
        self.conversation_embedding_service = conversation_embedding_service
        self.rag_evaluation_service = rag_evaluation_service
        
        # LLM providers registry
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._default_provider_id: Optional[str] = None
        
        # Active chat contexts
        self._chat_contexts: Dict[str, ChatContext] = {}
        
        # Response callbacks
        self._response_callbacks: Dict[str, List[Callable]] = {}
        self._context_callbacks: Dict[str, List[Callable]] = {}
        
        # Performance tracking
        self._response_count = 0
        self._avg_response_time = 0.0
        
        # Logger
        self.logger = logger.bind(component="RAGIntegrationService")
        
        # Initialize components flag
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the RAG integration service."""
        try:
            # Initialize core services
            if self.rag_service:
                await self.rag_service.initialize()
            if self.retrieval_service:
                await self.retrieval_service.initialize()
            if self.embedding_service:
                await self.embedding_service.initialize()
            if self.vector_store:
                await self.vector_store.initialize()
            if self.conversation_embedding_service:
                await self.conversation_embedding_service.initialize()
                
            # Register for configuration changes
            self.rag_config_service.add_change_listener(self._on_config_changed)
            
            # Set up event listeners
            await self._setup_event_listeners()
            
            self._initialized = True
            self.logger.info("RAG integration service initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG integration service: {e}")
            raise
            
    async def cleanup(self) -> None:
        """Cleanup the RAG integration service."""
        # Cleanup services
        services = [
            self.rag_service,
            self.retrieval_service, 
            self.embedding_service,
            self.vector_store,
            self.conversation_embedding_service
        ]
        
        for service in services:
            if service and hasattr(service, 'cleanup'):
                try:
                    await service.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up service: {e}")
                    
        self._chat_contexts.clear()
        self._response_callbacks.clear()
        self._context_callbacks.clear()
        
        self.logger.info("RAG integration service cleaned up")
        
    def register_provider(self, provider_id: str, provider: BaseLLMProvider) -> None:
        """Register an LLM provider."""
        self._providers[provider_id] = provider
        
        # Register with RAG service if available
        if self.rag_service:
            self.rag_service.register_provider(provider_id, provider)
            
        # Set as default if first provider
        if self._default_provider_id is None:
            self._default_provider_id = provider_id
            if self.rag_service:
                self.rag_service.set_default_provider(provider_id)
                
        self.logger.info(f"Registered LLM provider: {provider_id}")
        
    def set_default_provider(self, provider_id: str) -> None:
        """Set the default LLM provider."""
        if provider_id not in self._providers:
            raise ValueError(f"Provider {provider_id} not registered")
            
        self._default_provider_id = provider_id
        if self.rag_service:
            self.rag_service.set_default_provider(provider_id)
            
        self.logger.info(f"Default provider set to: {provider_id}")
        
    async def create_chat_context(
        self, 
        conversation_id: str, 
        user_id: Optional[str] = None,
        assistant_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        **kwargs
    ) -> ChatContext:
        """Create a chat context for a conversation."""
        context = ChatContext(
            conversation_id=conversation_id,
            user_id=user_id,
            assistant_id=assistant_id,
            provider_id=provider_id or self._default_provider_id,
            rag_enabled=kwargs.get('rag_enabled', True),
            rag_mode=kwargs.get('rag_mode', RAGMode.HYBRID)
        )
        
        self._chat_contexts[conversation_id] = context
        
        # Initialize callbacks lists
        self._response_callbacks[conversation_id] = []
        self._context_callbacks[conversation_id] = []
        
        self.logger.debug(f"Created chat context for conversation: {conversation_id}")
        return context
        
    def get_chat_context(self, conversation_id: str) -> Optional[ChatContext]:
        """Get chat context for a conversation."""
        return self._chat_contexts.get(conversation_id)
        
    async def process_message(
        self, 
        conversation_id: str, 
        user_message: str,
        **kwargs
    ) -> RAGResponse:
        """
        Process a user message and generate RAG-enhanced response.
        
        Args:
            conversation_id: ID of the conversation
            user_message: User's message text
            **kwargs: Additional parameters
            
        Returns:
            RAG response with enhanced content and context
        """
        if not self._initialized:
            raise RuntimeError("RAG integration service not initialized")
            
        start_time = time.time()
        
        try:
            # Get or create chat context
            context = self._chat_contexts.get(conversation_id)
            if not context:
                context = await self.create_chat_context(conversation_id)
                
            # Store user message
            await self._store_user_message(conversation_id, user_message)
            
            # Check if RAG is enabled
            if not context.rag_enabled or not self.rag_service:
                # Fall back to direct LLM call
                return await self._generate_direct_response(context, user_message)
                
            # Emit processing started event
            await self.event_bus.emit("rag_processing_started", {
                "conversation_id": conversation_id,
                "user_message": user_message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Create RAG request
            rag_request = RAGRequest(
                query=user_message,
                conversation_id=conversation_id,
                mode=context.rag_mode,
                provider_id=context.provider_id,
                model_name=context.model_name,
                config=self.rag_config_service.get_config().generation,
                user_id=context.user_id,
                session_metadata={
                    "assistant_id": context.assistant_id,
                    "ui_context": "chat"
                }
            )
            
            # Generate RAG response
            rag_response = await self.rag_service.generate_response(rag_request)
            
            # Update metrics
            total_time = int((time.time() - start_time) * 1000)
            self._update_metrics(total_time)
            
            # Create response metrics
            metrics = ResponseMetrics(
                retrieval_time_ms=rag_response.retrieval_time_ms,
                generation_time_ms=rag_response.generation_time_ms,
                total_time_ms=total_time,
                sources_count=len(rag_response.retrieved_sources),
                context_relevance=rag_response.context_relevance_score,
                response_length=len(rag_response.content)
            )
            
            # Evaluate RAG response if evaluation service is available
            evaluation = None
            if self.rag_evaluation_service and hasattr(rag_response, 'retrieved_sources'):
                try:
                    # Create retrieval context for evaluation
                    retrieval_context = RetrievalContext(
                        document_chunks=getattr(rag_response, 'retrieved_chunks', []),
                        similarity_scores=getattr(rag_response, 'similarity_scores', []),
                        retrieval_query=user_message,
                        total_retrieved=len(rag_response.retrieved_sources),
                        retrieval_time=rag_response.retrieval_time_ms / 1000.0
                    )
                    
                    # Evaluate the response
                    evaluation = await self.rag_evaluation_service.evaluate_response(
                        query=user_message,
                        response=rag_response.content,
                        retrieval_context=retrieval_context,
                        response_time=total_time / 1000.0,
                        token_usage=getattr(rag_response, 'token_usage', 0)
                    )
                    
                    # Add evaluation to response metadata
                    if not hasattr(rag_response, 'metadata') or rag_response.metadata is None:
                        rag_response.metadata = {}
                    rag_response.metadata['evaluation'] = evaluation.to_dict()
                    
                    self.logger.info(f"Response evaluated with overall score: {evaluation.overall_score:.3f}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate RAG response: {e}")
            
            # Store assistant response (with evaluation metadata)
            await self._store_assistant_message(
                conversation_id,
                rag_response.content,
                rag_response.metadata
            )
            
            # Emit processing completed event
            await self.event_bus.emit("rag_processing_completed", {
                "conversation_id": conversation_id,
                "response_length": len(rag_response.content),
                "sources_count": len(rag_response.retrieved_sources),
                "metrics": metrics.__dict__,
                "timestamp": datetime.now().isoformat()
            })
            
            # Notify callbacks
            await self._notify_response_callbacks(conversation_id, rag_response, metrics)
            await self._notify_context_callbacks(conversation_id, rag_response.retrieved_sources)
            
            self.logger.info(
                f"Processed message for conversation {conversation_id} "
                f"in {total_time}ms with {len(rag_response.retrieved_sources)} sources"
            )
            
            return rag_response
            
        except Exception as e:
            self.logger.error(f"Failed to process message: {e}")
            
            # Emit error event
            await self.event_bus.emit("rag_processing_error", {
                "conversation_id": conversation_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            raise
            
    async def process_message_streaming(
        self,
        conversation_id: str,
        user_message: str,
        response_callback: Callable[[str], None],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Process a message with streaming response generation.
        
        Args:
            conversation_id: ID of the conversation
            user_message: User's message text
            response_callback: Callback for response chunks
            **kwargs: Additional parameters
            
        Yields:
            Response chunks as they're generated
        """
        if not self._initialized:
            raise RuntimeError("RAG integration service not initialized")
            
        try:
            # Get chat context
            context = self._chat_contexts.get(conversation_id)
            if not context:
                context = await self.create_chat_context(conversation_id)
                
            # Store user message
            await self._store_user_message(conversation_id, user_message)
            
            # Check if RAG is enabled and streaming is supported
            if not context.rag_enabled or not self.rag_service:
                # Fall back to direct streaming
                async for chunk in self._generate_direct_response_streaming(context, user_message):
                    response_callback(chunk)
                    yield chunk
                return
                
            # Create RAG request
            rag_request = RAGRequest(
                query=user_message,
                conversation_id=conversation_id,
                mode=context.rag_mode,
                provider_id=context.provider_id,
                model_name=context.model_name,
                config=self.rag_config_service.get_config().generation
            )
            
            # Stream RAG response
            full_response = ""
            async for chunk in self.rag_service.generate_streaming_response(rag_request):
                full_response += chunk
                response_callback(chunk)
                yield chunk
                
            # Store complete response
            await self._store_assistant_message(conversation_id, full_response)
            
        except Exception as e:
            self.logger.error(f"Failed to process streaming message: {e}")
            error_chunk = f"Error: {str(e)}"
            response_callback(error_chunk)
            yield error_chunk
            
    async def _generate_direct_response(
        self, 
        context: ChatContext, 
        user_message: str
    ) -> RAGResponse:
        """Generate response without RAG (direct LLM call)."""
        if not context.provider_id or context.provider_id not in self._providers:
            raise ValueError("No valid provider available")
            
        provider = self._providers[context.provider_id]
        
        # Simple direct call to LLM
        from ..providers.base import ChatCompletionRequest, ChatMessage
        
        messages = [ChatMessage(role="user", content=user_message)]
        request = ChatCompletionRequest(
            messages=messages,
            model=context.model_name or provider.get_default_model(),
            max_tokens=self.rag_config_service.get_config().generation.max_response_tokens,
            temperature=self.rag_config_service.get_config().generation.temperature
        )
        
        response = await provider.create_chat_completion(request)
        
        if not response.choices:
            raise ValueError("No response from provider")
            
        # Create RAG response format
        return RAGResponse(
            content=response.choices[0].message.content,
            conversation_id=context.conversation_id,
            model_used=context.model_name or provider.get_default_model(),
            retrieved_sources=[],
            context_used="",
            retrieval_time_ms=0,
            generation_time_ms=100,  # Estimated
            total_time_ms=100,
            context_relevance_score=0.0,
            response_confidence=0.8
        )
        
    async def _generate_direct_response_streaming(
        self, 
        context: ChatContext, 
        user_message: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response without RAG."""
        if not context.provider_id or context.provider_id not in self._providers:
            yield "Error: No valid provider available"
            return
            
        provider = self._providers[context.provider_id]
        
        # Check if provider supports streaming
        if not hasattr(provider, 'create_chat_completion_stream'):
            # Fall back to non-streaming
            response = await self._generate_direct_response(context, user_message)
            yield response.content
            return
            
        from ..providers.base import ChatCompletionRequest, ChatMessage
        
        messages = [ChatMessage(role="user", content=user_message)]
        request = ChatCompletionRequest(
            messages=messages,
            model=context.model_name or provider.get_default_model(),
            max_tokens=self.rag_config_service.get_config().generation.max_response_tokens,
            temperature=self.rag_config_service.get_config().generation.temperature,
            stream=True
        )
        
        try:
            async for chunk in provider.create_chat_completion_stream(request):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error: {str(e)}"
            
    async def _store_user_message(self, conversation_id: str, content: str) -> None:
        """Store user message in database."""
        try:
            async with self.database_manager.get_session() as session:
                message = Message(
                    conversation_id=conversation_id,
                    role="user",
                    content=content,
                    created_at=datetime.now()
                )
                session.add(message)
                await session.commit()
        except Exception as e:
            self.logger.warning(f"Failed to store user message: {e}")
            
    async def _store_assistant_message(
        self, 
        conversation_id: str, 
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store assistant message in database."""
        try:
            async with self.database_manager.get_session() as session:
                message = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=content,
                    metadata=metadata or {},
                    created_at=datetime.now()
                )
                session.add(message)
                await session.commit()
        except Exception as e:
            self.logger.warning(f"Failed to store assistant message: {e}")
            
    def add_response_callback(
        self, 
        conversation_id: str, 
        callback: Callable[[RAGResponse, ResponseMetrics], None]
    ) -> None:
        """Add callback for response generation events."""
        if conversation_id not in self._response_callbacks:
            self._response_callbacks[conversation_id] = []
        self._response_callbacks[conversation_id].append(callback)
        
    def add_context_callback(
        self, 
        conversation_id: str, 
        callback: Callable[[List], None]
    ) -> None:
        """Add callback for context update events."""
        if conversation_id not in self._context_callbacks:
            self._context_callbacks[conversation_id] = []
        self._context_callbacks[conversation_id].append(callback)
        
    async def _notify_response_callbacks(
        self, 
        conversation_id: str, 
        response: RAGResponse,
        metrics: ResponseMetrics
    ) -> None:
        """Notify response callbacks."""
        callbacks = self._response_callbacks.get(conversation_id, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(response, metrics)
                else:
                    callback(response, metrics)
            except Exception as e:
                self.logger.warning(f"Response callback failed: {e}")
                
    async def _notify_context_callbacks(
        self, 
        conversation_id: str, 
        sources: List
    ) -> None:
        """Notify context callbacks."""
        callbacks = self._context_callbacks.get(conversation_id, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(sources)
                else:
                    callback(sources)
            except Exception as e:
                self.logger.warning(f"Context callback failed: {e}")
                
    async def _setup_event_listeners(self) -> None:
        """Set up event listeners."""
        # Listen for document processing events
        await self.event_bus.on("document_processed", self._on_document_processed)
        await self.event_bus.on("conversation_embedded", self._on_conversation_embedded)
        
    async def _on_document_processed(self, event_data: Dict[str, Any]) -> None:
        """Handle document processed event."""
        document_id = event_data.get("document_id")
        self.logger.debug(f"Document processed: {document_id}")
        
        # Emit event for UI updates
        await self.event_bus.emit("rag_index_updated", {
            "type": "document",
            "id": document_id,
            "timestamp": datetime.now().isoformat()
        })
        
    async def _on_conversation_embedded(self, event_data: Dict[str, Any]) -> None:
        """Handle conversation embedded event."""
        conversation_id = event_data.get("conversation_id")
        self.logger.debug(f"Conversation embedded: {conversation_id}")
        
        # Emit event for UI updates
        await self.event_bus.emit("rag_index_updated", {
            "type": "conversation",
            "id": conversation_id,
            "timestamp": datetime.now().isoformat()
        })
        
    async def _on_config_changed(self, old_config, new_config) -> None:
        """Handle configuration changes."""
        self.logger.info("RAG configuration changed, updating services")
        
        # Update services with new configuration
        # This could involve reinitializing components if needed
        await self.event_bus.emit("rag_config_applied", {
            "timestamp": datetime.now().isoformat()
        })
        
    def _update_metrics(self, response_time: int) -> None:
        """Update performance metrics."""
        self._response_count += 1
        alpha = 2.0 / (self._response_count + 1)
        self._avg_response_time = alpha * response_time + (1 - alpha) * self._avg_response_time
        
    def get_service_stats(self) -> Dict[str, Any]:
        """Get integration service statistics."""
        return {
            "initialized": self._initialized,
            "active_conversations": len(self._chat_contexts),
            "registered_providers": list(self._providers.keys()),
            "default_provider": self._default_provider_id,
            "response_count": self._response_count,
            "avg_response_time_ms": round(self._avg_response_time, 2),
            "rag_enabled_conversations": len([
                c for c in self._chat_contexts.values() if c.rag_enabled
            ])
        }
        
    async def enable_rag_for_conversation(self, conversation_id: str, enabled: bool = True) -> None:
        """Enable or disable RAG for a specific conversation."""
        context = self._chat_contexts.get(conversation_id)
        if context:
            context.rag_enabled = enabled
            self.logger.info(f"RAG {'enabled' if enabled else 'disabled'} for conversation {conversation_id}")
            
    async def set_rag_mode_for_conversation(self, conversation_id: str, mode: RAGMode) -> None:
        """Set RAG mode for a specific conversation."""
        context = self._chat_contexts.get(conversation_id)
        if context:
            context.rag_mode = mode
            self.logger.info(f"RAG mode set to {mode.value} for conversation {conversation_id}")
            
    def get_conversation_context_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get context summary for a conversation."""
        context = self._chat_contexts.get(conversation_id)
        if not context:
            return {}
            
        return {
            "conversation_id": conversation_id,
            "rag_enabled": context.rag_enabled,
            "rag_mode": context.rag_mode.value,
            "provider_id": context.provider_id,
            "model_name": context.model_name,
            "assistant_id": context.assistant_id
        }

    async def get_text_context(self, query: str) -> str:
        """(Simulation) Gets text context for a given query."""
        self.logger.info(f"Retrieving text context for query: '{query}'")
        await asyncio.sleep(0.5)  # Simulate retrieval time
        return f"Simulated text context related to '{query}'."