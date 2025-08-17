"""
RAG (Retrieval-Augmented Generation) service for Kate LLM Client.

This service orchestrates the complete RAG workflow, combining semantic retrieval
with LLM generation to provide contextually enhanced responses.
"""

import asyncio
import time
from typing import List, Dict, Optional, Any, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from ..core.events import EventBus
from ..database.manager import DatabaseManager
from ..database.models import Conversation, Message, RAGSession
from ..providers.base import BaseLLMProvider, ChatMessage, ChatCompletionRequest
from .retrieval_service import RetrievalService, RetrievalQuery, RetrievalResult, RetrievalMode
from .embedding_service import EmbeddingService


class RAGMode(Enum):
    """Different RAG operation modes."""
    STANDARD = "standard"  # Standard RAG with retrieval + generation
    CONVERSATIONAL = "conversational"  # Context-aware with conversation history
    FOCUSED = "focused"  # Document-focused retrieval
    HYBRID = "hybrid"  # Multiple retrieval strategies combined


class ContextStrategy(Enum):
    """Strategies for incorporating retrieved context."""
    PREPEND = "prepend"  # Add context before user query
    STRUCTURED = "structured"  # Use structured format with sections
    INLINE = "inline"  # Weave context into conversation naturally
    CITATION = "citation"  # Include citations and references


@dataclass
class RAGConfig:
    """Configuration for RAG operations."""
    # Retrieval settings
    max_retrieved_chunks: int = 10
    min_similarity_score: float = 0.7
    context_window_size: int = 4000  # Max characters for context
    
    # Generation settings
    max_response_tokens: int = 2000
    temperature: float = 0.7
    include_citations: bool = True
    
    # Context formatting
    context_strategy: ContextStrategy = ContextStrategy.STRUCTURED
    context_separator: str = "\n---\n"
    citation_format: str = "[{source}]"
    
    # Performance settings
    enable_streaming: bool = True
    timeout_seconds: int = 30
    max_concurrent_retrievals: int = 3
    
    # Quality control
    enable_response_validation: bool = True
    min_context_relevance: float = 0.6
    enable_fact_checking: bool = False


@dataclass
class RAGRequest:
    """Request for RAG-enhanced generation."""
    query: str
    conversation_id: str
    mode: RAGMode = RAGMode.STANDARD
    config: Optional[RAGConfig] = None
    
    # Override settings
    provider_id: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    
    # Context control
    document_filters: Optional[Dict[str, Any]] = None
    time_range: Optional[Tuple[str, str]] = None
    exclude_sources: Optional[List[str]] = None
    
    # Metadata
    user_id: Optional[str] = None
    session_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextSource:
    """Information about a context source."""
    content: str
    source_type: str  # 'document' or 'conversation'
    title: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    citation: Optional[str] = None


@dataclass
class RAGResponse:
    """Response from RAG-enhanced generation."""
    content: str
    conversation_id: str
    model_used: str
    
    # Context information
    retrieved_sources: List[ContextSource]
    context_used: str
    retrieval_time_ms: int
    generation_time_ms: int
    total_time_ms: int
    
    # Quality metrics
    context_relevance_score: float
    response_confidence: float
    
    # Session tracking
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGService:
    """
    Advanced RAG service orchestrating retrieval and generation.
    
    Combines semantic search with LLM generation to provide
    contextually enhanced, factually grounded responses.
    """
    
    def __init__(
        self,
        database_manager: DatabaseManager,
        retrieval_service: RetrievalService,
        embedding_service: EmbeddingService,
        event_bus: EventBus,
        config: Optional[RAGConfig] = None
    ):
        self.database_manager = database_manager
        self.retrieval_service = retrieval_service
        self.embedding_service = embedding_service
        self.event_bus = event_bus
        self.config = config or RAGConfig()
        
        # Provider registry for multi-provider support
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._default_provider_id: Optional[str] = None
        
        # Session tracking
        self._active_sessions: Dict[str, RAGSession] = {}
        
        # Performance monitoring
        self._request_count = 0
        self._avg_retrieval_time = 0.0
        self._avg_generation_time = 0.0
        
        # Logger
        self.logger = logger.bind(component="RAGService")
        
    async def initialize(self) -> None:
        """Initialize the RAG service."""
        await self.retrieval_service.initialize()
        self.logger.info("RAG service initialized")
        
    async def cleanup(self) -> None:
        """Cleanup RAG service resources."""
        await self.retrieval_service.cleanup()
        self._active_sessions.clear()
        self.logger.info("RAG service cleaned up")
        
    def register_provider(self, provider_id: str, provider: BaseLLMProvider) -> None:
        """Register an LLM provider."""
        self._providers[provider_id] = provider
        if self._default_provider_id is None:
            self._default_provider_id = provider_id
        self.logger.info(f"Registered provider: {provider_id}")
        
    def set_default_provider(self, provider_id: str) -> None:
        """Set the default LLM provider."""
        if provider_id not in self._providers:
            raise ValueError(f"Provider {provider_id} not registered")
        self._default_provider_id = provider_id
        self.logger.info(f"Default provider set to: {provider_id}")
        
    async def generate_response(self, request: RAGRequest) -> RAGResponse:
        """
        Generate RAG-enhanced response.
        
        Args:
            request: RAG request with query and parameters
            
        Returns:
            Enhanced response with retrieved context
        """
        start_time = time.time()
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Create session
            session = await self._create_session(request)
            
            # Phase 1: Retrieval
            retrieval_start = time.time()
            retrieved_context = await self._retrieve_context(request)
            retrieval_time = int((time.time() - retrieval_start) * 1000)
            
            if not retrieved_context:
                self.logger.warning("No relevant context retrieved")
                
            # Phase 2: Context processing
            formatted_context, context_sources = await self._format_context(
                retrieved_context, request
            )
            
            # Phase 3: Generation
            generation_start = time.time()
            response_content = await self._generate_with_context(
                request, formatted_context
            )
            generation_time = int((time.time() - generation_start) * 1000)
            
            # Phase 4: Post-processing
            final_response = await self._post_process_response(
                response_content, context_sources, request
            )
            
            total_time = int((time.time() - start_time) * 1000)
            
            # Build response
            rag_response = RAGResponse(
                content=final_response,
                conversation_id=request.conversation_id,
                model_used=self._get_model_name(request),
                retrieved_sources=context_sources,
                context_used=formatted_context,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=total_time,
                context_relevance_score=self._calculate_context_relevance(retrieved_context),
                response_confidence=0.8,  # Placeholder - could be calculated
                session_id=session.id if session else None,
                metadata=request.session_metadata
            )
            
            # Update session and metrics
            await self._update_session(session, rag_response)
            self._update_metrics(retrieval_time, generation_time)
            
            # Emit event
            await self.event_bus.emit("rag_response_generated", {
                "conversation_id": request.conversation_id,
                "response_length": len(final_response),
                "sources_count": len(context_sources),
                "total_time_ms": total_time
            })
            
            self.logger.info(
                f"RAG response generated in {total_time}ms "
                f"(retrieval: {retrieval_time}ms, generation: {generation_time}ms)"
            )
            
            return rag_response
            
        except Exception as e:
            self.logger.error(f"RAG generation failed: {e}")
            raise
            
    async def generate_streaming_response(
        self, 
        request: RAGRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming RAG-enhanced response.
        
        Args:
            request: RAG request with query and parameters
            
        Yields:
            Response chunks as they're generated
        """
        if not self.config.enable_streaming:
            # Fallback to non-streaming
            response = await self.generate_response(request)
            yield response.content
            return
            
        try:
            # Phase 1: Retrieval (same as non-streaming)
            retrieved_context = await self._retrieve_context(request)
            formatted_context, _ = await self._format_context(retrieved_context, request)
            
            # Phase 2: Streaming generation
            provider = self._get_provider(request)
            
            # Build messages with context
            messages = await self._build_messages_with_context(request, formatted_context)
            
            completion_request = ChatCompletionRequest(
                messages=messages,
                model=request.model_name or provider.get_default_model(),
                max_tokens=request.max_tokens or self.config.max_response_tokens,
                temperature=request.temperature or self.config.temperature,
                stream=True
            )
            
            # Stream response
            async for chunk in provider.create_chat_completion_stream(completion_request):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"Streaming RAG generation failed: {e}")
            yield f"Error: {str(e)}"
            
    async def _retrieve_context(self, request: RAGRequest) -> List[RetrievalResult]:
        """Retrieve relevant context for the request."""
        # Determine retrieval mode based on RAG mode
        if request.mode == RAGMode.FOCUSED:
            retrieval_mode = RetrievalMode.DOCUMENTS_ONLY
        elif request.mode == RAGMode.CONVERSATIONAL:
            retrieval_mode = RetrievalMode.CONTEXTUAL
        else:
            retrieval_mode = RetrievalMode.HYBRID
            
        # Build retrieval query
        retrieval_query = RetrievalQuery(
            text=request.query,
            mode=retrieval_mode,
            max_results=self.config.max_retrieved_chunks,
            min_score=self.config.min_similarity_score,
            conversation_id=request.conversation_id,
            document_filters=request.document_filters,
            time_range=request.time_range,
            exclude_chunk_ids=set(request.exclude_sources or [])
        )
        
        # Perform retrieval
        results = await self.retrieval_service.retrieve(retrieval_query)
        
        return results
        
    async def _format_context(
        self, 
        retrieved_results: List[RetrievalResult],
        request: RAGRequest
    ) -> Tuple[str, List[ContextSource]]:
        """Format retrieved context for LLM consumption."""
        if not retrieved_results:
            return "", []
            
        context_sources = []
        context_parts = []
        
        effective_config = request.config or self.config
        current_length = 0
        
        for i, result in enumerate(retrieved_results):
            # Check if adding this context would exceed the window
            if current_length + len(result.content) > effective_config.context_window_size:
                break
                
            # Create context source
            citation = None
            if effective_config.include_citations:
                source_name = result.document_title or result.conversation_id or f"Source {i+1}"
                citation = effective_config.citation_format.format(source=source_name)
                
            context_source = ContextSource(
                content=result.content,
                source_type=result.source_type,
                title=result.document_title,
                score=result.score,
                metadata=result.metadata or {},
                citation=citation
            )
            context_sources.append(context_source)
            
            # Format context part
            if effective_config.context_strategy == ContextStrategy.STRUCTURED:
                context_part = f"Source: {context_source.title or 'Unknown'}\n{result.content}"
                if citation:
                    context_part += f"\n{citation}"
            elif effective_config.context_strategy == ContextStrategy.CITATION:
                context_part = f"{result.content} {citation or ''}"
            else:  # PREPEND or INLINE
                context_part = result.content
                
            context_parts.append(context_part)
            current_length += len(result.content)
            
        # Combine context parts
        if effective_config.context_strategy == ContextStrategy.STRUCTURED:
            formatted_context = f"\n{effective_config.context_separator}\n".join(context_parts)
        else:
            formatted_context = "\n\n".join(context_parts)
            
        return formatted_context, context_sources
        
    async def _generate_with_context(
        self, 
        request: RAGRequest, 
        formatted_context: str
    ) -> str:
        """Generate response using LLM with context."""
        provider = self._get_provider(request)
        
        # Build messages with context
        messages = await self._build_messages_with_context(request, formatted_context)
        
        # Create completion request
        completion_request = ChatCompletionRequest(
            messages=messages,
            model=request.model_name or provider.get_default_model(),
            max_tokens=request.max_tokens or self.config.max_response_tokens,
            temperature=request.temperature or self.config.temperature,
            stream=False
        )
        
        # Generate response
        response = await provider.create_chat_completion(completion_request)
        
        if not response.choices:
            raise ValueError("No response generated from LLM")
            
        return response.choices[0].message.content
        
    async def _build_messages_with_context(
        self, 
        request: RAGRequest, 
        formatted_context: str
    ) -> List[ChatMessage]:
        """Build chat messages incorporating retrieved context."""
        effective_config = request.config or self.config
        
        messages = []
        
        if formatted_context:
            if effective_config.context_strategy == ContextStrategy.PREPEND:
                # Add context before user query
                system_prompt = (
                    "You are a helpful assistant. Use the following context to answer "
                    "the user's question. If the context doesn't contain relevant information, "
                    "you may use your general knowledge but indicate when you're doing so.\n\n"
                    f"Context:\n{formatted_context}"
                )
                messages.append(ChatMessage(role="system", content=system_prompt))
                messages.append(ChatMessage(role="user", content=request.query))
                
            elif effective_config.context_strategy == ContextStrategy.STRUCTURED:
                # Structured format with clear sections
                system_prompt = (
                    "You are a helpful assistant. Answer the user's question using the "
                    "provided context. Cite sources when possible using the format [Source Name]."
                )
                messages.append(ChatMessage(role="system", content=system_prompt))
                
                user_message = (
                    f"Context:\n{formatted_context}\n\n"
                    f"Question: {request.query}"
                )
                messages.append(ChatMessage(role="user", content=user_message))
                
            else:  # INLINE or CITATION
                # Inline context integration
                enhanced_query = f"{request.query}\n\nRelevant context:\n{formatted_context}"
                messages.append(ChatMessage(role="user", content=enhanced_query))
        else:
            # No context available
            messages.append(ChatMessage(role="user", content=request.query))
            
        return messages
        
    async def _post_process_response(
        self, 
        response_content: str,
        context_sources: List[ContextSource],
        request: RAGRequest
    ) -> str:
        """Post-process the generated response."""
        processed_content = response_content.strip()
        
        # Add citations if enabled and not already present
        effective_config = request.config or self.config
        if (effective_config.include_citations and 
            effective_config.context_strategy == ContextStrategy.CITATION and
            context_sources):
            
            # Simple citation addition (could be more sophisticated)
            citations = [cs.citation for cs in context_sources if cs.citation]
            if citations:
                processed_content += f"\n\nSources: {', '.join(citations)}"
                
        # Validate response if enabled
        if effective_config.enable_response_validation:
            processed_content = await self._validate_response(
                processed_content, context_sources
            )
            
        return processed_content
        
    async def _validate_response(
        self, 
        response: str, 
        sources: List[ContextSource]
    ) -> str:
        """Validate response quality and consistency."""
        # This is a placeholder for response validation logic
        # In a real implementation, this could include:
        # - Fact checking against sources
        # - Consistency checking
        # - Hallucination detection
        # - Quality scoring
        
        return response
        
    def _get_provider(self, request: RAGRequest) -> BaseLLMProvider:
        """Get the appropriate LLM provider for the request."""
        provider_id = request.provider_id or self._default_provider_id
        
        if not provider_id or provider_id not in self._providers:
            raise ValueError(f"Provider {provider_id} not available")
            
        return self._providers[provider_id]
        
    def _get_model_name(self, request: RAGRequest) -> str:
        """Get the model name used for the request."""
        provider = self._get_provider(request)
        return request.model_name or provider.get_default_model()
        
    def _validate_request(self, request: RAGRequest) -> None:
        """Validate RAG request parameters."""
        if not request.query.strip():
            raise ValueError("Query cannot be empty")
            
        if not request.conversation_id:
            raise ValueError("Conversation ID is required")
            
        # Validate provider
        self._get_provider(request)
        
    async def _create_session(self, request: RAGRequest) -> Optional[RAGSession]:
        """Create RAG session for tracking."""
        try:
            async with self.database_manager.get_session() as session:
                rag_session = RAGSession(
                    conversation_id=request.conversation_id,
                    query=request.query,
                    mode=request.mode.value,
                    config=request.config.__dict__ if request.config else self.config.__dict__,
                    user_id=request.user_id,
                    metadata=request.session_metadata
                )
                
                session.add(rag_session)
                await session.commit()
                await session.refresh(rag_session)
                
                self._active_sessions[rag_session.id] = rag_session
                return rag_session
                
        except Exception as e:
            self.logger.warning(f"Failed to create RAG session: {e}")
            return None
            
    async def _update_session(
        self, 
        session: Optional[RAGSession], 
        response: RAGResponse
    ) -> None:
        """Update RAG session with response metrics."""
        if not session:
            return
            
        try:
            async with self.database_manager.get_session() as db_session:
                session.sources_count = len(response.retrieved_sources)
                session.retrieval_time_ms = response.retrieval_time_ms
                session.generation_time_ms = response.generation_time_ms
                session.context_relevance_score = response.context_relevance_score
                session.response_length = len(response.content)
                session.completed_at = asyncio.get_event_loop().time()
                
                db_session.add(session)
                await db_session.commit()
                
        except Exception as e:
            self.logger.warning(f"Failed to update RAG session: {e}")
            
    def _calculate_context_relevance(self, results: List[RetrievalResult]) -> float:
        """Calculate average relevance score of retrieved context."""
        if not results:
            return 0.0
            
        return sum(r.score for r in results) / len(results)
        
    def _update_metrics(self, retrieval_time: int, generation_time: int) -> None:
        """Update performance metrics."""
        self._request_count += 1
        
        # Running average calculation
        alpha = 2.0 / (self._request_count + 1)
        self._avg_retrieval_time = (
            alpha * retrieval_time + (1 - alpha) * self._avg_retrieval_time
        )
        self._avg_generation_time = (
            alpha * generation_time + (1 - alpha) * self._avg_generation_time
        )
        
    def get_service_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics."""
        return {
            "request_count": self._request_count,
            "avg_retrieval_time_ms": round(self._avg_retrieval_time, 2),
            "avg_generation_time_ms": round(self._avg_generation_time, 2),
            "active_sessions": len(self._active_sessions),
            "registered_providers": list(self._providers.keys()),
            "default_provider": self._default_provider_id,
            "config": {
                "max_retrieved_chunks": self.config.max_retrieved_chunks,
                "context_window_size": self.config.context_window_size,
                "enable_streaming": self.config.enable_streaming,
                "include_citations": self.config.include_citations
            }
        }