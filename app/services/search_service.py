"""
Enhanced search service for Kate LLM Client with semantic search capabilities.

Provides comprehensive search functionality combining traditional text search
with semantic similarity search using embeddings for documents and conversations.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.orm import selectinload

from ..core.events import EventBus, SearchCompletedEvent, SearchStartedEvent
from ..database.manager import DatabaseManager
from ..database.models import (
    ChunkEmbedding,
    Conversation,
    ConversationEmbedding,
    Document,
    DocumentChunk,
    Message,
)
from .retrieval_service import RetrievalMode, RetrievalQuery, RetrievalService
from .vector_store import VectorStore

# Lazy import to avoid hanging during startup
_EmbeddingService = None

def _get_embedding_service():
    global _EmbeddingService
    if _EmbeddingService is None:
        from .embedding_service import EmbeddingService
        _EmbeddingService = EmbeddingService
    return _EmbeddingService


class SearchMode(Enum):
    """Different search modes available."""
    TEXT_ONLY = "text_only"  # Traditional text search
    SEMANTIC_ONLY = "semantic_only"  # Embedding-based semantic search
    HYBRID = "hybrid"  # Combination of text and semantic search
    CONTEXTUAL = "contextual"  # Context-aware semantic search


class SearchScope(Enum):
    """Search scope options."""
    ALL = "all"  # Search everything
    CONVERSATIONS = "conversations"  # Only conversations
    DOCUMENTS = "documents"  # Only documents
    MESSAGES = "messages"  # Only messages
    CURRENT_CONVERSATION = "current_conversation"  # Current conversation only


class ResultType(Enum):
    """Types of search results."""
    CONVERSATION = "conversation"
    MESSAGE = "message"
    DOCUMENT = "document"
    DOCUMENT_CHUNK = "document_chunk"
    CONVERSATION_SEGMENT = "conversation_segment"


@dataclass
class SearchFilter:
    """Search filtering options."""
    date_range: Optional[Tuple[datetime, datetime]] = None
    conversation_ids: Optional[List[str]] = None
    document_types: Optional[List[str]] = None
    min_score: float = 0.0
    max_results: int = 50
    include_metadata: bool = True


@dataclass
class SearchResult:
    """Enhanced search result with semantic information."""
    result_id: str
    result_type: ResultType
    title: str
    content: str
    snippet: str  # Highlighted excerpt
    score: float
    semantic_score: Optional[float] = None
    text_score: Optional[float] = None
    created_at: Optional[datetime] = None
    
    # Context information
    conversation_id: Optional[str] = None
    document_id: Optional[str] = None
    message_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    highlights: List[str] = field(default_factory=list)
    related_results: List[str] = field(default_factory=list)


@dataclass
class SearchRequest:
    """Comprehensive search request."""
    query: str
    mode: SearchMode = SearchMode.HYBRID
    scope: SearchScope = SearchScope.ALL
    filters: Optional[SearchFilter] = None
    enable_query_expansion: bool = True
    enable_reranking: bool = True
    context_conversation_id: Optional[str] = None


class SearchService:
    """
    Enhanced search service with semantic capabilities.
    
    Provides comprehensive search functionality combining traditional text search
    with semantic similarity search using embeddings.
    """
    
    def __init__(
        self,
        database_manager: DatabaseManager,
        event_bus: EventBus,
        embedding_service = None,
        retrieval_service: Optional[RetrievalService] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.database_manager = database_manager
        self.event_bus = event_bus
        self.embedding_service = embedding_service
        self.retrieval_service = retrieval_service
        self.vector_store = vector_store
        self.logger = logger.bind(component="SearchService")
        
        # Search configuration
        self.max_snippet_length = 200
        self.highlight_prefix = "<mark>"
        self.highlight_suffix = "</mark>"
        self.semantic_weight = 0.7  # Weight for semantic vs text search in hybrid mode
        
        # Performance tracking
        self._search_count = 0
        self._avg_search_time = 0.0
        
    async def initialize(self) -> None:
        """Initialize the search service."""
        self.logger.info("Enhanced search service initialized")
        
    async def cleanup(self) -> None:
        """Cleanup search service resources."""
        self.logger.info("Search service cleaned up")
        
    async def search(self, request: SearchRequest) -> List[SearchResult]:
        """
        Perform comprehensive search with the specified parameters.
        
        Args:
            request: Search request with query and options
            
        Returns:
            List of ranked search results
        """
        start_time = time.time()
        
        try:
            self.logger.debug(
                f"Performing {request.mode.value} search: '{request.query}' "
                f"(scope: {request.scope.value})"
            )
            
            # Emit search started event
            await self.event_bus.emit("search_started", {
                "query": request.query,
                "mode": request.mode.value,
                "scope": request.scope.value
            })
            
            # Expand query if enabled
            if request.enable_query_expansion:
                expanded_query = await self._expand_query(request)
                if expanded_query != request.query:
                    self.logger.debug(f"Expanded query: '{expanded_query}'")
                    request.query = expanded_query
            
            # Perform search based on mode
            if request.mode == SearchMode.TEXT_ONLY:
                results = await self._text_search(request)
            elif request.mode == SearchMode.SEMANTIC_ONLY:
                results = await self._semantic_search(request)
            elif request.mode == SearchMode.CONTEXTUAL:
                results = await self._contextual_search(request)
            else:  # HYBRID
                results = await self._hybrid_search(request)
                
            # Post-process results
            results = await self._post_process_results(results, request)
            
            # Apply filters
            if request.filters:
                results = self._apply_filters(results, request.filters)
                
            # Re-rank if enabled
            if request.enable_reranking:
                results = await self._rerank_results(results, request)
                
            # Limit results
            max_results = request.filters.max_results if request.filters else 50
            results = results[:max_results]
            
            search_time = time.time() - start_time
            self._update_metrics(search_time)
            
            self.logger.info(
                f"Search completed: {len(results)} results in {search_time:.2f}s"
            )
            
            # Emit search completed event
            await self.event_bus.emit("search_completed", {
                "query": request.query,
                "results_count": len(results),
                "search_time": search_time,
                "mode": request.mode.value
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
            
    async def search_conversations(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search conversations (backward compatibility).
        
        Args:
            query: Search query string
            **kwargs: Additional search options
            
        Returns:
            List of matching conversations
        """
        request = SearchRequest(
            query=query,
            scope=SearchScope.CONVERSATIONS,
            mode=kwargs.get('mode', SearchMode.HYBRID)
        )
        
        results = await self.search(request)
        
        # Convert to legacy format
        return [
            {
                "id": result.conversation_id,
                "title": result.title,
                "content": result.content,
                "score": result.score,
                "created_at": result.created_at.isoformat() if result.created_at else None,
                "snippet": result.snippet
            }
            for result in results
            if result.result_type == ResultType.CONVERSATION
        ]
        
    async def search_messages(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search messages (backward compatibility).
        
        Args:
            query: Search query string
            conversation_id: Optional conversation ID to limit search
            **kwargs: Additional search options
            
        Returns:
            List of matching messages
        """
        request = SearchRequest(
            query=query,
            scope=SearchScope.CURRENT_CONVERSATION if conversation_id else SearchScope.MESSAGES,
            mode=kwargs.get('mode', SearchMode.HYBRID),
            context_conversation_id=conversation_id
        )
        
        if conversation_id:
            filters = SearchFilter(conversation_ids=[conversation_id])
            request.filters = filters
        
        results = await self.search(request)
        
        # Convert to legacy format
        return [
            {
                "id": result.message_id,
                "content": result.content,
                "score": result.score,
                "conversation_id": result.conversation_id,
                "created_at": result.created_at.isoformat() if result.created_at else None,
                "snippet": result.snippet
            }
            for result in results
            if result.result_type == ResultType.MESSAGE
        ]
        
    async def search_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search documents and document chunks.
        
        Args:
            query: Search query string
            **kwargs: Additional search options
            
        Returns:
            List of matching documents and chunks
        """
        request = SearchRequest(
            query=query,
            scope=SearchScope.DOCUMENTS,
            mode=kwargs.get('mode', SearchMode.HYBRID)
        )
        
        results = await self.search(request)
        
        # Convert to dict format
        return [
            {
                "id": result.document_id or result.result_id,
                "title": result.title,
                "content": result.content,
                "score": result.score,
                "type": result.result_type.value,
                "created_at": result.created_at.isoformat() if result.created_at else None,
                "snippet": result.snippet,
                "metadata": result.metadata
            }
            for result in results
            if result.result_type in [ResultType.DOCUMENT, ResultType.DOCUMENT_CHUNK]
        ]
        
    async def _expand_query(self, request: SearchRequest) -> str:
        """Expand the search query with related terms."""
        # Simple query expansion - could be enhanced with synonym expansion,
        # related terms from conversation context, etc.
        query = request.query.strip()
        
        # Add context from current conversation if available
        if request.context_conversation_id and self.database_manager:
            try:
                async with self.database_manager.get_session() as session:
                    # Get recent messages for context
                    result = await session.execute(
                        select(Message)
                        .where(Message.conversation_id == request.context_conversation_id)
                        .order_by(desc(Message.created_at))
                        .limit(5)
                    )
                    recent_messages = result.scalars().all()
                    
                    if recent_messages:
                        # Extract key terms from recent context (simplified)
                        context_words = set()
                        for msg in recent_messages:
                            words = msg.content.lower().split()
                            # Add important words (longer than 3 chars, not common words)
                            context_words.update([
                                w for w in words
                                if len(w) > 3 and w not in ['this', 'that', 'with', 'from', 'they', 'have', 'been']
                            ])
                        
                        # Add a few context terms to query
                        if context_words:
                            context_terms = list(context_words)[:3]
                            query += " " + " ".join(context_terms)
                            
            except Exception as e:
                self.logger.warning(f"Query expansion failed: {e}")
                
        return query
        
    async def _text_search(self, request: SearchRequest) -> List[SearchResult]:
        """Perform traditional text-based search."""
        results = []
        
        async with self.database_manager.get_session() as session:
            # Search conversations
            if request.scope in [SearchScope.ALL, SearchScope.CONVERSATIONS]:
                conv_results = await self._search_conversations_text(session, request)
                results.extend(conv_results)
                
            # Search messages
            if request.scope in [SearchScope.ALL, SearchScope.MESSAGES, SearchScope.CURRENT_CONVERSATION]:
                msg_results = await self._search_messages_text(session, request)
                results.extend(msg_results)
                
            # Search documents
            if request.scope in [SearchScope.ALL, SearchScope.DOCUMENTS]:
                doc_results = await self._search_documents_text(session, request)
                results.extend(doc_results)
                
        return results
        
    async def _semantic_search(self, request: SearchRequest) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        if not self.retrieval_service:
            self.logger.warning("Retrieval service not available for semantic search")
            return await self._text_search(request)
            
        results = []
        
        # Determine retrieval mode based on scope
        if request.scope == SearchScope.CONVERSATIONS:
            retrieval_mode = RetrievalMode.CONVERSATIONS_ONLY
        elif request.scope == SearchScope.DOCUMENTS:
            retrieval_mode = RetrievalMode.DOCUMENTS_ONLY
        else:
            retrieval_mode = RetrievalMode.HYBRID
            
        # Create retrieval query
        retrieval_query = RetrievalQuery(
            text=request.query,
            mode=retrieval_mode,
            max_results=request.filters.max_results if request.filters else 50,
            min_score=request.filters.min_score if request.filters else 0.5,
            conversation_id=request.context_conversation_id
        )
        
        # Perform retrieval
        retrieval_results = await self.retrieval_service.retrieve(retrieval_query)
        
        # Convert to search results
        for ret_result in retrieval_results:
            search_result = SearchResult(
                result_id=ret_result.chunk_id,
                result_type=ResultType.DOCUMENT_CHUNK if ret_result.source_type == "document" else ResultType.CONVERSATION_SEGMENT,
                title=ret_result.document_title or f"Conversation {ret_result.conversation_id}",
                content=ret_result.content,
                snippet=self._create_snippet(ret_result.content, request.query),
                score=ret_result.score,
                semantic_score=ret_result.score,
                created_at=datetime.fromisoformat(ret_result.created_at) if ret_result.created_at else None,
                conversation_id=ret_result.conversation_id,
                document_id=ret_result.document_id,
                metadata=ret_result.metadata or {}
            )
            results.append(search_result)
            
        return results
        
    async def _hybrid_search(self, request: SearchRequest) -> List[SearchResult]:
        """Perform hybrid search combining text and semantic search."""
        # Get results from both methods
        text_results = await self._text_search(request)
        semantic_results = await self._semantic_search(request)
        
        # Combine and re-score results
        combined_results = {}
        
        # Add text results
        for result in text_results:
            result.text_score = result.score
            result.score = result.score * (1 - self.semantic_weight)
            combined_results[result.result_id] = result
            
        # Add semantic results
        for result in semantic_results:
            if result.result_id in combined_results:
                # Combine scores
                existing = combined_results[result.result_id]
                existing.semantic_score = result.score
                existing.score += result.score * self.semantic_weight
                # Use better snippet
                if result.score > existing.text_score:
                    existing.snippet = result.snippet
            else:
                result.score = result.score * self.semantic_weight
                combined_results[result.result_id] = result
                
        # Sort by combined score
        results = list(combined_results.values())
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
        
    async def _contextual_search(self, request: SearchRequest) -> List[SearchResult]:
        """Perform context-aware semantic search."""
        # This is similar to semantic search but with enhanced context awareness
        return await self._semantic_search(request)
        
    async def _search_conversations_text(self, session, request: SearchRequest) -> List[SearchResult]:
        """Search conversations using text matching."""
        query_lower = request.query.lower()
        
        # Build query with filters
        stmt = select(Conversation).where(
            or_(
                func.lower(Conversation.title).contains(query_lower),
                func.lower(Conversation.description).contains(query_lower) if hasattr(Conversation, 'description') else False
            )
        )
        
        # Apply conversation ID filter if specified
        if request.filters and request.filters.conversation_ids:
            stmt = stmt.where(Conversation.id.in_(request.filters.conversation_ids))
            
        # Apply date filter
        if request.filters and request.filters.date_range:
            start_date, end_date = request.filters.date_range
            stmt = stmt.where(and_(
                Conversation.created_at >= start_date,
                Conversation.created_at <= end_date
            ))
            
        stmt = stmt.order_by(desc(Conversation.updated_at)).limit(25)
        
        result = await session.execute(stmt)
        conversations = result.scalars().all()
        
        search_results = []
        for conv in conversations:
            # Calculate simple text similarity score
            title_score = self._calculate_text_similarity(request.query, conv.title)
            desc_score = 0
            if hasattr(conv, 'description') and conv.description:
                desc_score = self._calculate_text_similarity(request.query, conv.description)
                
            score = max(title_score, desc_score)
            
            if score > 0.1:  # Minimum relevance threshold
                search_result = SearchResult(
                    result_id=conv.id,
                    result_type=ResultType.CONVERSATION,
                    title=conv.title,
                    content=getattr(conv, 'description', '') or conv.title,
                    snippet=self._create_snippet(conv.title, request.query),
                    score=score,
                    text_score=score,
                    created_at=conv.created_at,
                    conversation_id=conv.id,
                    metadata={"message_count": getattr(conv, 'message_count', 0)}
                )
                search_results.append(search_result)
                
        return search_results
        
    async def _search_messages_text(self, session, request: SearchRequest) -> List[SearchResult]:
        """Search messages using text matching."""
        query_lower = request.query.lower()
        
        # Build query
        stmt = select(Message).options(selectinload(Message.conversation)).where(
            func.lower(Message.content).contains(query_lower)
        )
        
        # Apply conversation filter
        if request.context_conversation_id:
            stmt = stmt.where(Message.conversation_id == request.context_conversation_id)
        elif request.filters and request.filters.conversation_ids:
            stmt = stmt.where(Message.conversation_id.in_(request.filters.conversation_ids))
            
        # Apply date filter
        if request.filters and request.filters.date_range:
            start_date, end_date = request.filters.date_range
            stmt = stmt.where(and_(
                Message.created_at >= start_date,
                Message.created_at <= end_date
            ))
            
        stmt = stmt.order_by(desc(Message.created_at)).limit(50)
        
        result = await session.execute(stmt)
        messages = result.scalars().all()
        
        search_results = []
        for msg in messages:
            score = self._calculate_text_similarity(request.query, msg.content)
            
            if score > 0.1:
                search_result = SearchResult(
                    result_id=msg.id,
                    result_type=ResultType.MESSAGE,
                    title=f"Message in {msg.conversation.title if msg.conversation else 'Unknown'}",
                    content=msg.content,
                    snippet=self._create_snippet(msg.content, request.query),
                    score=score,
                    text_score=score,
                    created_at=msg.created_at,
                    conversation_id=msg.conversation_id,
                    message_id=msg.id,
                    metadata={"role": msg.role}
                )
                search_results.append(search_result)
                
        return search_results
        
    async def _search_documents_text(self, session, request: SearchRequest) -> List[SearchResult]:
        """Search documents using text matching."""
        query_lower = request.query.lower()
        
        # Search documents
        doc_stmt = select(Document).where(
            or_(
                func.lower(Document.title).contains(query_lower),
                func.lower(Document.content).contains(query_lower)
            )
        )
        
        if request.filters and request.filters.document_types:
            doc_stmt = doc_stmt.where(Document.file_type.in_(request.filters.document_types))
            
        doc_result = await session.execute(doc_stmt.limit(25))
        documents = doc_result.scalars().all()
        
        search_results = []
        for doc in documents:
            title_score = self._calculate_text_similarity(request.query, doc.title)
            content_score = self._calculate_text_similarity(request.query, doc.content or "")
            score = max(title_score, content_score)
            
            if score > 0.1:
                search_result = SearchResult(
                    result_id=doc.id,
                    result_type=ResultType.DOCUMENT,
                    title=doc.title,
                    content=doc.content or "",
                    snippet=self._create_snippet(doc.content or doc.title, request.query),
                    score=score,
                    text_score=score,
                    created_at=doc.created_at,
                    document_id=doc.id,
                    metadata={
                        "file_type": doc.file_type,
                        "file_size": doc.file_size,
                        "processed": doc.processed
                    }
                )
                search_results.append(search_result)
                
        return search_results
        
    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """Calculate simple text similarity score."""
        if not query or not text:
            return 0.0
            
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Exact match gets highest score
        if query_lower in text_lower:
            return 1.0
            
        # Word overlap scoring
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        
        if not query_words:
            return 0.0
            
        overlap = len(query_words.intersection(text_words))
        return overlap / len(query_words)
        
    def _create_snippet(self, content: str, query: str) -> str:
        """Create a highlighted snippet from content."""
        if not content:
            return ""
            
        # Find the best position for the snippet
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Try to find query in content
        pos = content_lower.find(query_lower)
        if pos == -1:
            # If not found, look for individual words
            words = query_lower.split()
            for word in words:
                pos = content_lower.find(word)
                if pos != -1:
                    break
                    
        if pos == -1:
            # No match found, return beginning
            pos = 0
            
        # Extract snippet around the position
        start = max(0, pos - self.max_snippet_length // 2)
        end = min(len(content), start + self.max_snippet_length)
        
        snippet = content[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
            
        # Highlight query terms (simple implementation)
        for word in query.split():
            if len(word) > 2:  # Only highlight longer words
                snippet = snippet.replace(
                    word,
                    f"{self.highlight_prefix}{word}{self.highlight_suffix}"
                )
                snippet = snippet.replace(
                    word.lower(),
                    f"{self.highlight_prefix}{word.lower()}{self.highlight_suffix}"
                )
                snippet = snippet.replace(
                    word.capitalize(),
                    f"{self.highlight_prefix}{word.capitalize()}{self.highlight_suffix}"
                )
                
        return snippet
        
    async def _post_process_results(self, results: List[SearchResult], request: SearchRequest) -> List[SearchResult]:
        """Post-process search results."""
        # Remove duplicates
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result.result_id not in seen_ids:
                seen_ids.add(result.result_id)
                unique_results.append(result)
                
        # Sort by score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results
        
    def _apply_filters(self, results: List[SearchResult], filters: SearchFilter) -> List[SearchResult]:
        """Apply search filters to results."""
        filtered_results = []
        
        for result in results:
            # Score filter
            if result.score < filters.min_score:
                continue
                
            # Date range filter
            if filters.date_range and result.created_at:
                start_date, end_date = filters.date_range
                if not (start_date <= result.created_at <= end_date):
                    continue
                    
            # Conversation ID filter
            if filters.conversation_ids and result.conversation_id:
                if result.conversation_id not in filters.conversation_ids:
                    continue
                    
            filtered_results.append(result)
            
        return filtered_results
        
    async def _rerank_results(self, results: List[SearchResult], request: SearchRequest) -> List[SearchResult]:
        """Re-rank search results using advanced scoring."""
        # Simple re-ranking based on result type preferences
        # In a more sophisticated implementation, this could use learning-to-rank models
        
        type_weights = {
            ResultType.MESSAGE: 1.0,
            ResultType.CONVERSATION: 0.9,
            ResultType.DOCUMENT_CHUNK: 0.8,
            ResultType.DOCUMENT: 0.7,
            ResultType.CONVERSATION_SEGMENT: 0.6
        }
        
        for result in results:
            type_weight = type_weights.get(result.result_type, 1.0)
            result.score *= type_weight
            
        # Sort by adjusted score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
        
    def _update_metrics(self, search_time: float) -> None:
        """Update search performance metrics."""
        self._search_count += 1
        alpha = 2.0 / (self._search_count + 1)
        self._avg_search_time = alpha * search_time + (1 - alpha) * self._avg_search_time
        
    def get_service_stats(self) -> Dict[str, Any]:
        """Get search service statistics."""
        return {
            "search_count": self._search_count,
            "avg_search_time": round(self._avg_search_time, 3),
            "has_semantic_search": self.retrieval_service is not None,
            "has_embedding_service": self.embedding_service is not None,
            "config": {
                "max_snippet_length": self.max_snippet_length,
                "semantic_weight": self.semantic_weight
            }
        }