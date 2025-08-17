"""
Retrieval service for Kate LLM Client.

This service provides intelligent retrieval capabilities for RAG,
combining semantic search across documents and conversation history
with advanced ranking and filtering strategies.
"""

import asyncio
import time
from typing import List, Dict, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass
from enum import Enum

from loguru import logger
from sqlalchemy import select, and_, or_

from ..core.events import EventBus
from ..database.manager import DatabaseManager
from ..database.models import Document, DocumentChunk, Conversation, Message
from .vector_store import VectorStore, SearchResult
from .embedding_service import EmbeddingService


class RetrievalMode(Enum):
    """Different retrieval modes."""
    DOCUMENTS_ONLY = "documents_only"
    CONVERSATIONS_ONLY = "conversations_only"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"  # Uses conversation context


class RankingStrategy(Enum):
    """Different ranking strategies."""
    SIMILARITY_ONLY = "similarity_only"
    RECENCY_BOOST = "recency_boost"
    POPULARITY_BOOST = "popularity_boost"
    HYBRID_RANK = "hybrid_rank"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations."""
    max_results: int = 10
    min_similarity_score: float = 0.7
    chunk_overlap_threshold: float = 0.8  # For deduplication
    recency_weight: float = 0.1  # Weight for recency in ranking
    popularity_weight: float = 0.05  # Weight for popularity in ranking
    conversation_context_window: int = 5  # Number of recent messages for context
    enable_query_expansion: bool = True
    enable_reranking: bool = True
    max_chunk_length: int = 2000  # Max characters per chunk in results


@dataclass
class RetrievalQuery:
    """A retrieval query with context and parameters."""
    text: str
    mode: RetrievalMode = RetrievalMode.HYBRID
    ranking_strategy: RankingStrategy = RankingStrategy.HYBRID_RANK
    max_results: Optional[int] = None
    min_score: Optional[float] = None
    conversation_id: Optional[str] = None
    document_filters: Optional[Dict[str, Any]] = None
    time_range: Optional[Tuple[str, str]] = None  # (start_date, end_date)
    exclude_chunk_ids: Optional[Set[str]] = None


@dataclass
class RetrievalResult:
    """Enhanced search result with additional context."""
    chunk_id: str
    content: str
    score: float
    source_type: str  # 'document' or 'conversation'
    document_title: Optional[str] = None
    document_id: Optional[str] = None
    conversation_id: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    ranking_factors: Optional[Dict[str, float]] = None


class RetrievalService:
    """
    Advanced retrieval service for RAG applications.
    
    Provides intelligent search across documents and conversations
    with sophisticated ranking, filtering, and context-aware capabilities.
    """
    
    def __init__(
        self,
        database_manager: DatabaseManager,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        event_bus: EventBus,
        config: Optional[RetrievalConfig] = None
    ):
        self.database_manager = database_manager
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.event_bus = event_bus
        self.config = config or RetrievalConfig()
        
        # Query cache for performance
        self._query_cache: Dict[str, List[RetrievalResult]] = {}
        self._cache_ttl_seconds = 300  # 5 minutes
        self._cache_timestamps: Dict[str, float] = {}
        
        # Logger
        self.logger = logger.bind(component="RetrievalService")
        
    async def initialize(self) -> None:
        """Initialize the retrieval service."""
        self.logger.info("Retrieval service initialized")
        
    async def cleanup(self) -> None:
        """Cleanup retrieval service resources."""
        self._query_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Retrieval service cleaned up")
        
    async def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Perform intelligent retrieval based on query parameters.
        
        Args:
            query: Retrieval query with parameters
            
        Returns:
            List of ranked retrieval results
        """
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._get_cache_key(query)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Retrieved {len(cached_result)} results from cache")
                return cached_result
                
            # Process query
            processed_query = await self._process_query(query)
            
            # Retrieve based on mode
            if query.mode == RetrievalMode.DOCUMENTS_ONLY:
                results = await self._retrieve_documents(processed_query)
            elif query.mode == RetrievalMode.CONVERSATIONS_ONLY:
                results = await self._retrieve_conversations(processed_query)
            elif query.mode == RetrievalMode.CONTEXTUAL:
                results = await self._retrieve_contextual(processed_query)
            else:  # HYBRID
                results = await self._retrieve_hybrid(processed_query)
                
            # Post-process results
            final_results = await self._post_process_results(results, query)
            
            # Cache results
            self._cache_result(cache_key, final_results)
            
            retrieval_time = int((time.time() - start_time) * 1000)
            self.logger.info(
                f"Retrieved {len(final_results)} results in {retrieval_time}ms "
                f"(mode: {query.mode.value})"
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            raise
            
    async def _process_query(self, query: RetrievalQuery) -> RetrievalQuery:
        """
        Process and enhance the query.
        
        Args:
            query: Original query
            
        Returns:
            Processed query
        """
        processed_text = query.text.strip()
        
        # Query expansion if enabled
        if self.config.enable_query_expansion:
            processed_text = await self._expand_query(processed_text, query.conversation_id)
            
        # Create processed query
        processed_query = RetrievalQuery(
            text=processed_text,
            mode=query.mode,
            ranking_strategy=query.ranking_strategy,
            max_results=query.max_results or self.config.max_results,
            min_score=query.min_score or self.config.min_similarity_score,
            conversation_id=query.conversation_id,
            document_filters=query.document_filters,
            time_range=query.time_range,
            exclude_chunk_ids=query.exclude_chunk_ids or set()
        )
        
        return processed_query
        
    async def _expand_query(self, query_text: str, conversation_id: Optional[str]) -> str:
        """
        Expand query with conversation context if available.
        
        Args:
            query_text: Original query text
            conversation_id: Optional conversation ID for context
            
        Returns:
            Expanded query text
        """
        if not conversation_id:
            return query_text
            
        try:
            # Get recent conversation context
            async with self.database_manager.get_session() as session:
                recent_messages = await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at.desc())
                    .limit(self.config.conversation_context_window)
                )
                
                context_messages = recent_messages.scalars().all()
                
            if context_messages:
                # Extract key terms from recent conversation
                context_text = " ".join([msg.content for msg in context_messages])
                
                # Simple expansion: add relevant context terms
                # In a more sophisticated implementation, this could use
                # semantic similarity or keyword extraction
                expanded_query = f"{query_text} {context_text[:200]}"
                return expanded_query
                
        except Exception as e:
            self.logger.warning(f"Query expansion failed: {e}")
            
        return query_text
        
    async def _retrieve_documents(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve from documents only."""
        search_results = await self.vector_store.search_documents(
            query=query.text,
            max_results=query.max_results * 2,  # Get more for filtering
            score_threshold=query.min_score,
            filter_metadata=query.document_filters
        )
        
        return await self._convert_search_results(search_results, "document")
        
    async def _retrieve_conversations(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve from conversations only."""
        search_results = await self.vector_store.search_conversations(
            query=query.text,
            max_results=query.max_results * 2,
            score_threshold=query.min_score
        )
        
        return await self._convert_search_results(search_results, "conversation")
        
    async def _retrieve_contextual(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve with conversation context awareness."""
        # Get conversation-specific results first
        conv_results = await self._retrieve_conversations(query)
        
        # Filter for current conversation if specified
        if query.conversation_id:
            conv_results = [
                r for r in conv_results 
                if r.conversation_id == query.conversation_id
            ]
            
        # Get document results
        doc_results = await self._retrieve_documents(query)
        
        # Combine and rerank with contextual scoring
        all_results = conv_results + doc_results
        
        return all_results
        
    async def _retrieve_hybrid(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve using hybrid approach (documents + conversations)."""
        # Get results from both sources in parallel
        doc_task = self._retrieve_documents(query)
        conv_task = self._retrieve_conversations(query)
        
        doc_results, conv_results = await asyncio.gather(doc_task, conv_task)
        
        # Combine results
        all_results = doc_results + conv_results
        
        return all_results
        
    async def _convert_search_results(
        self, 
        search_results: List[SearchResult], 
        source_type: str
    ) -> List[RetrievalResult]:
        """Convert vector search results to retrieval results."""
        retrieval_results = []
        
        async with self.database_manager.get_session() as session:
            for result in search_results:
                try:
                    # Get additional metadata from database
                    if source_type == "document":
                        # Get chunk and document info
                        chunk = await session.get(DocumentChunk, result.chunk_id)
                        if chunk:
                            document = await session.get(Document, chunk.document_id)
                            
                            retrieval_result = RetrievalResult(
                                chunk_id=result.chunk_id,
                                content=result.content,
                                score=result.score,
                                source_type=source_type,
                                document_title=document.title if document else None,
                                document_id=chunk.document_id,
                                created_at=chunk.created_at.isoformat() if chunk.created_at else None,
                                metadata=result.metadata
                            )
                            
                            retrieval_results.append(retrieval_result)
                            
                    elif source_type == "conversation":
                        # Get conversation info
                        conversation_id = result.metadata.get("conversation_id")
                        if conversation_id:
                            conversation = await session.get(Conversation, conversation_id)
                            
                            retrieval_result = RetrievalResult(
                                chunk_id=result.chunk_id,
                                content=result.content,
                                score=result.score,
                                source_type=source_type,
                                conversation_id=conversation_id,
                                document_title=conversation.title if conversation else None,
                                created_at=conversation.created_at.isoformat() if conversation else None,
                                metadata=result.metadata
                            )
                            
                            retrieval_results.append(retrieval_result)
                            
                except Exception as e:
                    self.logger.warning(f"Failed to process search result {result.chunk_id}: {e}")
                    continue
                    
        return retrieval_results
        
    async def _post_process_results(
        self, 
        results: List[RetrievalResult], 
        query: RetrievalQuery
    ) -> List[RetrievalResult]:
        """Post-process and rank retrieval results."""
        # Filter excluded chunks
        if query.exclude_chunk_ids:
            results = [r for r in results if r.chunk_id not in query.exclude_chunk_ids]
            
        # Apply time range filtering
        if query.time_range:
            results = self._filter_by_time_range(results, query.time_range)
            
        # Remove duplicates and overlapping chunks
        results = await self._deduplicate_results(results)
        
        # Truncate long content
        results = self._truncate_content(results)
        
        # Re-rank if enabled
        if self.config.enable_reranking:
            results = await self._rerank_results(results, query)
        else:
            # Simple ranking by similarity score
            results.sort(key=lambda x: x.score, reverse=True)
            
        # Limit to max results
        max_results = query.max_results or self.config.max_results
        results = results[:max_results]
        
        return results
        
    def _filter_by_time_range(
        self, 
        results: List[RetrievalResult], 
        time_range: Tuple[str, str]
    ) -> List[RetrievalResult]:
        """Filter results by time range."""
        from datetime import datetime
        
        start_date, end_date = time_range
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        filtered_results = []
        for result in results:
            if result.created_at:
                try:
                    created_dt = datetime.fromisoformat(result.created_at)
                    if start_dt <= created_dt <= end_dt:
                        filtered_results.append(result)
                except ValueError:
                    # Invalid date format, include result
                    filtered_results.append(result)
            else:
                # No date info, include result
                filtered_results.append(result)
                
        return filtered_results
        
    async def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate and highly overlapping results."""
        if not results:
            return results
            
        # Simple deduplication by content similarity
        deduplicated = []
        
        for result in results:
            is_duplicate = False
            
            for existing in deduplicated:
                # Check content overlap
                content_similarity = self._calculate_content_overlap(
                    result.content, existing.content
                )
                
                if content_similarity > self.config.chunk_overlap_threshold:
                    # Keep the one with higher score
                    if result.score > existing.score:
                        deduplicated.remove(existing)
                        deduplicated.append(result)
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                deduplicated.append(result)
                
        return deduplicated
        
    def _calculate_content_overlap(self, content1: str, content2: str) -> float:
        """Calculate overlap between two content strings."""
        if not content1 or not content2:
            return 0.0
            
        # Simple word-based overlap calculation
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
        
    def _truncate_content(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Truncate content that exceeds maximum length."""
        for result in results:
            if len(result.content) > self.config.max_chunk_length:
                result.content = result.content[:self.config.max_chunk_length] + "..."
                
        return results
        
    async def _rerank_results(
        self, 
        results: List[RetrievalResult], 
        query: RetrievalQuery
    ) -> List[RetrievalResult]:
        """Re-rank results using the specified strategy."""
        if not results:
            return results
            
        if query.ranking_strategy == RankingStrategy.SIMILARITY_ONLY:
            # Already sorted by similarity
            return results
            
        # Calculate ranking factors for each result
        for result in results:
            ranking_factors = {"similarity": result.score}
            
            # Add recency factor
            if query.ranking_strategy in [RankingStrategy.RECENCY_BOOST, RankingStrategy.HYBRID_RANK]:
                recency_score = self._calculate_recency_score(result)
                ranking_factors["recency"] = recency_score
                
            # Add popularity factor (could be based on access frequency)
            if query.ranking_strategy in [RankingStrategy.POPULARITY_BOOST, RankingStrategy.HYBRID_RANK]:
                popularity_score = self._calculate_popularity_score(result)
                ranking_factors["popularity"] = popularity_score
                
            result.ranking_factors = ranking_factors
            
        # Calculate final scores and sort
        if query.ranking_strategy == RankingStrategy.HYBRID_RANK:
            for result in results:
                factors = result.ranking_factors
                final_score = (
                    factors["similarity"] +
                    factors.get("recency", 0) * self.config.recency_weight +
                    factors.get("popularity", 0) * self.config.popularity_weight
                )
                result.score = final_score
                
        elif query.ranking_strategy == RankingStrategy.RECENCY_BOOST:
            for result in results:
                factors = result.ranking_factors
                result.score = factors["similarity"] + factors.get("recency", 0) * 0.2
                
        elif query.ranking_strategy == RankingStrategy.POPULARITY_BOOST:
            for result in results:
                factors = result.ranking_factors
                result.score = factors["similarity"] + factors.get("popularity", 0) * 0.2
                
        # Sort by final score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
        
    def _calculate_recency_score(self, result: RetrievalResult) -> float:
        """Calculate recency score (0-1, where 1 is most recent)."""
        if not result.created_at:
            return 0.0
            
        try:
            from datetime import datetime, timezone
            
            created_dt = datetime.fromisoformat(result.created_at)
            now = datetime.now(timezone.utc)
            
            # Calculate days since creation
            days_ago = (now - created_dt).days
            
            # Exponential decay: score decreases with age
            # Recent items (< 7 days) get higher scores
            if days_ago < 7:
                return 1.0
            elif days_ago < 30:
                return 0.7
            elif days_ago < 90:
                return 0.4
            else:
                return 0.1
                
        except Exception:
            return 0.0
            
    def _calculate_popularity_score(self, result: RetrievalResult) -> float:
        """Calculate popularity score based on usage patterns."""
        # This is a placeholder - in a real implementation, this could be based on:
        # - Number of times the chunk has been retrieved
        # - User engagement metrics
        # - Citation frequency
        # For now, return a default score
        return 0.5
        
    def _get_cache_key(self, query: RetrievalQuery) -> str:
        """Generate cache key for query."""
        key_parts = [
            query.text,
            query.mode.value,
            query.ranking_strategy.value,
            str(query.max_results),
            str(query.min_score),
            query.conversation_id or "",
            str(hash(frozenset(query.document_filters.items()) if query.document_filters else frozenset())),
            str(query.time_range),
            str(hash(frozenset(query.exclude_chunk_ids) if query.exclude_chunk_ids else frozenset()))
        ]
        
        return "|".join(key_parts)
        
    def _get_cached_result(self, cache_key: str) -> Optional[List[RetrievalResult]]:
        """Get cached result if still valid."""
        if cache_key in self._query_cache:
            cache_time = self._cache_timestamps.get(cache_key, 0)
            if time.time() - cache_time < self._cache_ttl_seconds:
                return self._query_cache[cache_key]
            else:
                # Remove expired cache entry
                del self._query_cache[cache_key]
                del self._cache_timestamps[cache_key]
                
        return None
        
    def _cache_result(self, cache_key: str, results: List[RetrievalResult]) -> None:
        """Cache retrieval results."""
        self._query_cache[cache_key] = results
        self._cache_timestamps[cache_key] = time.time()
        
        # Simple cache size management
        if len(self._query_cache) > 100:
            # Remove oldest entry
            oldest_key = min(self._cache_timestamps.keys(), key=self._cache_timestamps.get)
            del self._query_cache[oldest_key]
            del self._cache_timestamps[oldest_key]
            
    async def search_similar_chunks(
        self, 
        chunk_content: str, 
        max_results: int = 5,
        exclude_chunk_id: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Find chunks similar to the given content.
        
        Args:
            chunk_content: Content to find similar chunks for
            max_results: Maximum number of results
            exclude_chunk_id: Chunk ID to exclude from results
            
        Returns:
            List of similar chunks
        """
        exclude_set = {exclude_chunk_id} if exclude_chunk_id else set()
        
        query = RetrievalQuery(
            text=chunk_content,
            mode=RetrievalMode.DOCUMENTS_ONLY,
            max_results=max_results,
            exclude_chunk_ids=exclude_set
        )
        
        return await self.retrieve(query)
        
    async def get_conversation_context(
        self, 
        conversation_id: str, 
        query_text: str,
        max_results: int = 3
    ) -> List[RetrievalResult]:
        """
        Get relevant context from a specific conversation.
        
        Args:
            conversation_id: Conversation to search in
            query_text: Query to find relevant context for
            max_results: Maximum number of results
            
        Returns:
            List of relevant conversation context
        """
        query = RetrievalQuery(
            text=query_text,
            mode=RetrievalMode.CONVERSATIONS_ONLY,
            conversation_id=conversation_id,
            max_results=max_results
        )
        
        results = await self.retrieve(query)
        
        # Filter for the specific conversation
        return [r for r in results if r.conversation_id == conversation_id]
        
    def get_service_stats(self) -> Dict[str, Any]:
        """Get retrieval service statistics."""
        return {
            "cache_size": len(self._query_cache),
            "cache_hit_rate": "N/A",  # Would need to track hits/misses
            "config": {
                "max_results": self.config.max_results,
                "min_similarity_score": self.config.min_similarity_score,
                "chunk_overlap_threshold": self.config.chunk_overlap_threshold,
                "enable_query_expansion": self.config.enable_query_expansion,
                "enable_reranking": self.config.enable_reranking
            }
        }