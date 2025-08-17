"""
Conversation embedding service for Kate LLM Client.

This service processes conversation history into semantic chunks and embeddings
for historical context retrieval in RAG workflows.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from loguru import logger
from sqlalchemy import select, and_, or_, desc
from sqlalchemy.orm import selectinload

from ..core.events import EventBus
from ..database.manager import DatabaseManager
from ..database.models import Conversation, Message, ConversationEmbedding
from .embedding_service import EmbeddingService
from .vector_store import VectorStore


class ChunkingStrategy(Enum):
    """Different strategies for chunking conversations."""
    MESSAGE_BASED = "message_based"  # Each message is a chunk
    TURN_BASED = "turn_based"  # User-assistant exchanges
    TOPIC_BASED = "topic_based"  # Semantic topic boundaries
    TIME_BASED = "time_based"  # Time-window chunks
    SLIDING_WINDOW = "sliding_window"  # Overlapping windows


class ConversationSegmentType(Enum):
    """Types of conversation segments."""
    SINGLE_MESSAGE = "single_message"
    EXCHANGE = "exchange"  # User message + assistant response
    TOPIC_CLUSTER = "topic_cluster"  # Related messages on same topic
    TIME_WINDOW = "time_window"  # Messages within time window


@dataclass
class ConversationSegment:
    """A semantic chunk of conversation history."""
    segment_id: str
    conversation_id: str
    segment_type: ConversationSegmentType
    content: str
    participants: List[str]  # User, assistant roles
    message_ids: List[str]
    start_time: datetime
    end_time: datetime
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingJob:
    """A job for processing conversation embeddings."""
    job_id: str
    conversation_ids: List[str]
    strategy: ChunkingStrategy
    options: Dict[str, Any]
    created_at: datetime
    status: str = "pending"  # pending, processing, completed, failed
    progress: float = 0.0
    error_message: Optional[str] = None


class ConversationEmbeddingService:
    """
    Service for processing conversation history into semantic embeddings.
    
    Extracts meaningful segments from conversations and creates embeddings
    for semantic retrieval of historical context.
    """
    
    def __init__(
        self,
        database_manager: DatabaseManager,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        event_bus: EventBus
    ):
        self.database_manager = database_manager
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.event_bus = event_bus
        
        # Processing configuration
        self.default_chunk_size = 512  # tokens
        self.overlap_size = 64  # tokens for sliding window
        self.min_chunk_size = 50  # minimum tokens
        self.max_chunk_size = 2048  # maximum tokens
        
        # Active jobs tracking
        self._active_jobs: Dict[str, EmbeddingJob] = {}
        self._processing_lock = asyncio.Lock()
        
        # Performance metrics
        self._processed_conversations = 0
        self._processed_segments = 0
        self._avg_processing_time = 0.0
        
        # Logger
        self.logger = logger.bind(component="ConversationEmbeddingService")
        
    async def initialize(self) -> None:
        """Initialize the conversation embedding service."""
        await self.vector_store.initialize()
        self.logger.info("Conversation embedding service initialized")
        
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        self._active_jobs.clear()
        self.logger.info("Conversation embedding service cleaned up")
        
    async def process_conversation(
        self,
        conversation_id: str,
        strategy: ChunkingStrategy = ChunkingStrategy.TURN_BASED,
        options: Optional[Dict[str, Any]] = None
    ) -> List[ConversationSegment]:
        """
        Process a single conversation into semantic segments.
        
        Args:
            conversation_id: ID of conversation to process
            strategy: Chunking strategy to use
            options: Additional processing options
            
        Returns:
            List of conversation segments
        """
        start_time = time.time()
        options = options or {}
        
        try:
            # Load conversation with messages
            async with self.database_manager.get_session() as session:
                result = await session.execute(
                    select(Conversation)
                    .options(selectinload(Conversation.messages))
                    .where(Conversation.id == conversation_id)
                )
                conversation = result.scalar_one_or_none()
                
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} not found")
                    
                if not conversation.messages:
                    self.logger.warning(f"No messages found in conversation {conversation_id}")
                    return []
                    
            # Sort messages by timestamp
            messages = sorted(conversation.messages, key=lambda m: m.created_at)
            
            # Generate segments based on strategy
            if strategy == ChunkingStrategy.MESSAGE_BASED:
                segments = await self._chunk_by_messages(conversation, messages, options)
            elif strategy == ChunkingStrategy.TURN_BASED:
                segments = await self._chunk_by_turns(conversation, messages, options)
            elif strategy == ChunkingStrategy.TOPIC_BASED:
                segments = await self._chunk_by_topics(conversation, messages, options)
            elif strategy == ChunkingStrategy.TIME_BASED:
                segments = await self._chunk_by_time(conversation, messages, options)
            elif strategy == ChunkingStrategy.SLIDING_WINDOW:
                segments = await self._chunk_sliding_window(conversation, messages, options)
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy}")
                
            # Generate embeddings for segments
            embedded_segments = await self._embed_segments(segments)
            
            # Store embeddings in database and vector store
            await self._store_embeddings(embedded_segments)
            
            processing_time = time.time() - start_time
            self._update_metrics(1, len(segments), processing_time)
            
            self.logger.info(
                f"Processed conversation {conversation_id} into {len(segments)} segments "
                f"in {processing_time:.2f}s"
            )
            
            # Emit processing event
            await self.event_bus.emit("conversation_embedded", {
                "conversation_id": conversation_id,
                "segments_count": len(segments),
                "strategy": strategy.value,
                "processing_time": processing_time
            })
            
            return embedded_segments
            
        except Exception as e:
            self.logger.error(f"Failed to process conversation {conversation_id}: {e}")
            raise
            
    async def process_conversations_batch(
        self,
        conversation_ids: List[str],
        strategy: ChunkingStrategy = ChunkingStrategy.TURN_BASED,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process multiple conversations in batch.
        
        Args:
            conversation_ids: List of conversation IDs to process
            strategy: Chunking strategy to use
            options: Additional processing options
            
        Returns:
            Job ID for tracking progress
        """
        job_id = f"embed_job_{int(time.time())}"
        job = EmbeddingJob(
            job_id=job_id,
            conversation_ids=conversation_ids,
            strategy=strategy,
            options=options or {},
            created_at=datetime.now()
        )
        
        self._active_jobs[job_id] = job
        
        # Start background processing
        asyncio.create_task(self._process_batch_background(job))
        
        self.logger.info(f"Started batch embedding job {job_id} for {len(conversation_ids)} conversations")
        return job_id
        
    async def _process_batch_background(self, job: EmbeddingJob) -> None:
        """Process batch job in background."""
        async with self._processing_lock:
            try:
                job.status = "processing"
                total_conversations = len(job.conversation_ids)
                
                for i, conversation_id in enumerate(job.conversation_ids):
                    try:
                        await self.process_conversation(
                            conversation_id, job.strategy, job.options
                        )
                        
                        # Update progress
                        job.progress = (i + 1) / total_conversations
                        
                        # Emit progress event
                        await self.event_bus.emit("embedding_progress", {
                            "job_id": job.job_id,
                            "progress": job.progress,
                            "completed": i + 1,
                            "total": total_conversations
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process conversation {conversation_id}: {e}")
                        continue
                        
                job.status = "completed"
                job.progress = 1.0
                
                self.logger.info(f"Completed batch embedding job {job.job_id}")
                
            except Exception as e:
                job.status = "failed"
                job.error_message = str(e)
                self.logger.error(f"Batch embedding job {job.job_id} failed: {e}")
                
            finally:
                # Emit completion event
                await self.event_bus.emit("embedding_completed", {
                    "job_id": job.job_id,
                    "status": job.status,
                    "error": job.error_message
                })
                
    async def _chunk_by_messages(
        self,
        conversation: Conversation,
        messages: List[Message],
        options: Dict[str, Any]
    ) -> List[ConversationSegment]:
        """Chunk conversation by individual messages."""
        segments = []
        
        for i, message in enumerate(messages):
            # Skip very short messages
            if len(message.content.strip()) < self.min_chunk_size:
                continue
                
            segment = ConversationSegment(
                segment_id=f"{conversation.id}_msg_{i}",
                conversation_id=conversation.id,
                segment_type=ConversationSegmentType.SINGLE_MESSAGE,
                content=message.content,
                participants=[message.role],
                message_ids=[message.id],
                start_time=message.created_at,
                end_time=message.created_at,
                token_count=len(message.content.split()),  # Rough token count
                metadata={
                    "message_index": i,
                    "role": message.role,
                    "message_id": message.id
                }
            )
            
            segments.append(segment)
            
        return segments
        
    async def _chunk_by_turns(
        self,
        conversation: Conversation,
        messages: List[Message],
        options: Dict[str, Any]
    ) -> List[ConversationSegment]:
        """Chunk conversation by user-assistant turn exchanges."""
        segments = []
        current_turn = []
        turn_index = 0
        
        for message in messages:
            current_turn.append(message)
            
            # End turn when we have both user and assistant messages
            # or when role changes back to user
            if (len(current_turn) >= 2 and 
                any(m.role == "user" for m in current_turn) and
                any(m.role == "assistant" for m in current_turn)):
                
                # Create segment for this turn
                segment = await self._create_turn_segment(
                    conversation, current_turn, turn_index
                )
                if segment:
                    segments.append(segment)
                    
                # Start new turn with current message if it's a user message
                current_turn = [message] if message.role == "user" else []
                turn_index += 1
                
        # Handle remaining messages in final turn
        if current_turn:
            segment = await self._create_turn_segment(
                conversation, current_turn, turn_index
            )
            if segment:
                segments.append(segment)
                
        return segments
        
    async def _create_turn_segment(
        self,
        conversation: Conversation,
        turn_messages: List[Message],
        turn_index: int
    ) -> Optional[ConversationSegment]:
        """Create a segment from a conversation turn."""
        if not turn_messages:
            return None
            
        # Combine messages into single content
        content_parts = []
        participants = set()
        message_ids = []
        
        for message in turn_messages:
            role_prefix = "User" if message.role == "user" else "Assistant"
            content_parts.append(f"{role_prefix}: {message.content}")
            participants.add(message.role)
            message_ids.append(message.id)
            
        content = "\n".join(content_parts)
        token_count = len(content.split())
        
        # Skip if too short
        if token_count < self.min_chunk_size:
            return None
            
        return ConversationSegment(
            segment_id=f"{conversation.id}_turn_{turn_index}",
            conversation_id=conversation.id,
            segment_type=ConversationSegmentType.EXCHANGE,
            content=content,
            participants=list(participants),
            message_ids=message_ids,
            start_time=turn_messages[0].created_at,
            end_time=turn_messages[-1].created_at,
            token_count=token_count,
            metadata={
                "turn_index": turn_index,
                "message_count": len(turn_messages)
            }
        )
        
    async def _chunk_by_topics(
        self,
        conversation: Conversation,
        messages: List[Message],
        options: Dict[str, Any]
    ) -> List[ConversationSegment]:
        """Chunk conversation by semantic topics (simplified implementation)."""
        # This is a simplified topic-based chunking
        # In a real implementation, this could use topic modeling or semantic similarity
        
        segments = []
        current_chunk = []
        chunk_index = 0
        
        # Group messages by time proximity as a simple topic heuristic
        time_threshold = options.get("topic_time_threshold", 300)  # 5 minutes
        
        for i, message in enumerate(messages):
            if (current_chunk and 
                (message.created_at - current_chunk[-1].created_at).total_seconds() > time_threshold):
                
                # Create segment for current chunk
                segment = await self._create_topic_segment(
                    conversation, current_chunk, chunk_index
                )
                if segment:
                    segments.append(segment)
                    
                current_chunk = [message]
                chunk_index += 1
            else:
                current_chunk.append(message)
                
        # Handle final chunk
        if current_chunk:
            segment = await self._create_topic_segment(
                conversation, current_chunk, chunk_index
            )
            if segment:
                segments.append(segment)
                
        return segments
        
    async def _create_topic_segment(
        self,
        conversation: Conversation,
        chunk_messages: List[Message],
        chunk_index: int
    ) -> Optional[ConversationSegment]:
        """Create a segment from a topic-based chunk."""
        if not chunk_messages:
            return None
            
        # Combine messages
        content_parts = []
        participants = set()
        message_ids = []
        
        for message in chunk_messages:
            content_parts.append(message.content)
            participants.add(message.role)
            message_ids.append(message.id)
            
        content = "\n\n".join(content_parts)
        token_count = len(content.split())
        
        if token_count < self.min_chunk_size:
            return None
            
        return ConversationSegment(
            segment_id=f"{conversation.id}_topic_{chunk_index}",
            conversation_id=conversation.id,
            segment_type=ConversationSegmentType.TOPIC_CLUSTER,
            content=content,
            participants=list(participants),
            message_ids=message_ids,
            start_time=chunk_messages[0].created_at,
            end_time=chunk_messages[-1].created_at,
            token_count=token_count,
            metadata={
                "topic_index": chunk_index,
                "message_count": len(chunk_messages)
            }
        )
        
    async def _chunk_by_time(
        self,
        conversation: Conversation,
        messages: List[Message],
        options: Dict[str, Any]
    ) -> List[ConversationSegment]:
        """Chunk conversation by time windows."""
        segments = []
        window_minutes = options.get("time_window_minutes", 30)
        window_delta = timedelta(minutes=window_minutes)
        
        if not messages:
            return segments
            
        current_window_start = messages[0].created_at
        current_chunk = []
        chunk_index = 0
        
        for message in messages:
            # Check if message fits in current window
            if message.created_at <= current_window_start + window_delta:
                current_chunk.append(message)
            else:
                # Create segment for current window
                if current_chunk:
                    segment = await self._create_time_segment(
                        conversation, current_chunk, chunk_index, current_window_start
                    )
                    if segment:
                        segments.append(segment)
                        
                # Start new window
                current_window_start = message.created_at
                current_chunk = [message]
                chunk_index += 1
                
        # Handle final window
        if current_chunk:
            segment = await self._create_time_segment(
                conversation, current_chunk, chunk_index, current_window_start
            )
            if segment:
                segments.append(segment)
                
        return segments
        
    async def _create_time_segment(
        self,
        conversation: Conversation,
        window_messages: List[Message],
        chunk_index: int,
        window_start: datetime
    ) -> Optional[ConversationSegment]:
        """Create a segment from a time window."""
        if not window_messages:
            return None
            
        # Combine messages
        content_parts = []
        participants = set()
        message_ids = []
        
        for message in window_messages:
            role_prefix = "User" if message.role == "user" else "Assistant"
            content_parts.append(f"{role_prefix}: {message.content}")
            participants.add(message.role)
            message_ids.append(message.id)
            
        content = "\n".join(content_parts)
        token_count = len(content.split())
        
        if token_count < self.min_chunk_size:
            return None
            
        return ConversationSegment(
            segment_id=f"{conversation.id}_time_{chunk_index}",
            conversation_id=conversation.id,
            segment_type=ConversationSegmentType.TIME_WINDOW,
            content=content,
            participants=list(participants),
            message_ids=message_ids,
            start_time=window_messages[0].created_at,
            end_time=window_messages[-1].created_at,
            token_count=token_count,
            metadata={
                "time_index": chunk_index,
                "window_start": window_start.isoformat(),
                "message_count": len(window_messages)
            }
        )
        
    async def _chunk_sliding_window(
        self,
        conversation: Conversation,
        messages: List[Message],
        options: Dict[str, Any]
    ) -> List[ConversationSegment]:
        """Chunk conversation using sliding window with overlap."""
        segments = []
        window_size = options.get("window_size", 5)  # Number of messages
        overlap = options.get("overlap", 2)  # Number of overlapping messages
        step = window_size - overlap
        
        for i in range(0, len(messages), step):
            window_messages = messages[i:i + window_size]
            
            if len(window_messages) < 2:  # Skip very small windows
                continue
                
            segment = await self._create_window_segment(
                conversation, window_messages, i // step
            )
            if segment:
                segments.append(segment)
                
        return segments
        
    async def _create_window_segment(
        self,
        conversation: Conversation,
        window_messages: List[Message],
        window_index: int
    ) -> Optional[ConversationSegment]:
        """Create a segment from a sliding window."""
        if not window_messages:
            return None
            
        # Combine messages
        content_parts = []
        participants = set()
        message_ids = []
        
        for message in window_messages:
            role_prefix = "User" if message.role == "user" else "Assistant"
            content_parts.append(f"{role_prefix}: {message.content}")
            participants.add(message.role)
            message_ids.append(message.id)
            
        content = "\n".join(content_parts)
        token_count = len(content.split())
        
        if token_count < self.min_chunk_size:
            return None
            
        return ConversationSegment(
            segment_id=f"{conversation.id}_window_{window_index}",
            conversation_id=conversation.id,
            segment_type=ConversationSegmentType.TIME_WINDOW,
            content=content,
            participants=list(participants),
            message_ids=message_ids,
            start_time=window_messages[0].created_at,
            end_time=window_messages[-1].created_at,
            token_count=token_count,
            metadata={
                "window_index": window_index,
                "message_count": len(window_messages)
            }
        )
        
    async def _embed_segments(self, segments: List[ConversationSegment]) -> List[ConversationSegment]:
        """Generate embeddings for conversation segments."""
        if not segments:
            return segments
            
        # Extract content for embedding
        texts = [segment.content for segment in segments]
        
        # Generate embeddings
        embeddings = await self.embedding_service.embed_texts(texts)
        
        # Add embeddings to segments
        for segment, embedding in zip(segments, embeddings):
            segment.metadata["embedding"] = embedding
            
        return segments
        
    async def _store_embeddings(self, segments: List[ConversationSegment]) -> None:
        """Store conversation embeddings in database and vector store."""
        if not segments:
            return
            
        async with self.database_manager.get_session() as session:
            for segment in segments:
                # Store in database
                conv_embedding = ConversationEmbedding(
                    id=segment.segment_id,
                    conversation_id=segment.conversation_id,
                    content=segment.content,
                    embedding=segment.metadata.get("embedding"),
                    segment_type=segment.segment_type.value,
                    message_ids=segment.message_ids,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    token_count=segment.token_count,
                    metadata=segment.metadata
                )
                
                session.add(conv_embedding)
                
                # Store in vector store
                await self.vector_store.add_conversation_chunk(
                    chunk_id=segment.segment_id,
                    content=segment.content,
                    embedding=segment.metadata.get("embedding"),
                    metadata={
                        "conversation_id": segment.conversation_id,
                        "segment_type": segment.segment_type.value,
                        "participants": segment.participants,
                        "start_time": segment.start_time.isoformat(),
                        "end_time": segment.end_time.isoformat(),
                        "token_count": segment.token_count
                    }
                )
                
            await session.commit()
            
    def _update_metrics(self, conversations: int, segments: int, processing_time: float) -> None:
        """Update processing metrics."""
        self._processed_conversations += conversations
        self._processed_segments += segments
        
        # Running average of processing time
        if self._processed_conversations > 0:
            alpha = 2.0 / (self._processed_conversations + 1)
            self._avg_processing_time = (
                alpha * processing_time + (1 - alpha) * self._avg_processing_time
            )
            
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch processing job."""
        job = self._active_jobs.get(job_id)
        if not job:
            return None
            
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "total_conversations": len(job.conversation_ids),
            "created_at": job.created_at.isoformat(),
            "error_message": job.error_message
        }
        
    async def get_conversation_segments(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all segments for a conversation."""
        async with self.database_manager.get_session() as session:
            result = await session.execute(
                select(ConversationEmbedding)
                .where(ConversationEmbedding.conversation_id == conversation_id)
                .order_by(ConversationEmbedding.start_time)
            )
            
            embeddings = result.scalars().all()
            
            return [
                {
                    "segment_id": emb.id,
                    "content": emb.content,
                    "segment_type": emb.segment_type,
                    "start_time": emb.start_time.isoformat(),
                    "end_time": emb.end_time.isoformat(),
                    "token_count": emb.token_count,
                    "metadata": emb.metadata
                }
                for emb in embeddings
            ]
            
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "processed_conversations": self._processed_conversations,
            "processed_segments": self._processed_segments,
            "avg_processing_time": round(self._avg_processing_time, 2),
            "active_jobs": len(self._active_jobs),
            "config": {
                "default_chunk_size": self.default_chunk_size,
                "overlap_size": self.overlap_size,
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size
            }
        }