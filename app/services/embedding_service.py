"""
Embedding service for Kate LLM Client.

This service handles text-to-vector conversion using sentence-transformers
for semantic search and RAG functionality.
"""

import asyncio
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from ..core.events import EventBus
from ..database.manager import DatabaseManager
from ..database.models import ChunkEmbedding, ConversationEmbedding


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    text: str
    embedding: np.ndarray
    model_name: str
    processing_time_ms: int


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str
    dimension: int
    max_sequence_length: int
    batch_size: int = 32
    device: str = "cpu"
    normalize_embeddings: bool = True


class EmbeddingService:
    """
    Service for generating and managing text embeddings.
    
    Uses sentence-transformers for high-quality semantic embeddings
    suitable for RAG and semantic search applications.
    """
    
    # Available embedding models with their configurations
    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2": EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            max_sequence_length=256,
            batch_size=64,
            normalize_embeddings=True
        ),
        "all-mpnet-base-v2": EmbeddingConfig(
            model_name="all-mpnet-base-v2", 
            dimension=768,
            max_sequence_length=384,
            batch_size=32,
            normalize_embeddings=True
        ),
        "multi-qa-MiniLM-L6-cos-v1": EmbeddingConfig(
            model_name="multi-qa-MiniLM-L6-cos-v1",
            dimension=384,
            max_sequence_length=512,
            batch_size=64,
            normalize_embeddings=True
        ),
        "paraphrase-multilingual-MiniLM-L12-v2": EmbeddingConfig(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            dimension=384,
            max_sequence_length=128,
            batch_size=64,
            normalize_embeddings=True
        )
    }
    
    def __init__(
        self, 
        database_manager: DatabaseManager,
        event_bus: EventBus,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None
    ):
        self.database_manager = database_manager
        self.event_bus = event_bus
        self.model_name = model_name
        self.cache_dir = cache_dir or Path.home() / ".cache" / "kate" / "embeddings"
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model and configuration
        self.config = self.AVAILABLE_MODELS.get(model_name)
        if not self.config:
            raise ValueError(f"Unknown embedding model: {model_name}")
            
        self.model: Optional[SentenceTransformer] = None
        self._model_lock = asyncio.Lock()
        
        # In-memory cache for frequently used embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_size_limit = 1000
        
        # Logger
        self.logger = logger.bind(component="EmbeddingService")
        
    async def initialize(self) -> None:
        """Initialize the embedding service and load the model."""
        try:
            self.logger.info(f"Initializing embedding service with model: {self.model_name}")
            await self._load_model()
            self.logger.info("Embedding service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding service: {e}")
            raise
            
    async def cleanup(self) -> None:
        """Cleanup embedding service resources."""
        try:
            if self.model:
                # Clear model from memory
                self.model = None
                
            # Clear cache
            self._embedding_cache.clear()
            
            self.logger.info("Embedding service cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    async def _load_model(self) -> None:
        """Load the sentence transformer model."""
        async with self._model_lock:
            if self.model is None:
                # Load model in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, 
                    self._load_model_sync
                )
                
    def _load_model_sync(self) -> SentenceTransformer:
        """Synchronously load the sentence transformer model."""
        self.logger.info(f"Loading sentence transformer model: {self.model_name}")
        
        # Load model with specific device
        model = SentenceTransformer(
            self.model_name,
            device=self.config.device,
            cache_folder=str(self.cache_dir / "models")
        )
        
        return model
        
    async def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding result with vector and metadata
        """
        results = await self.embed_texts([text])
        return results[0]
        
    async def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding results
        """
        if not texts:
            return []
            
        try:
            import time
            start_time = time.time()
            
            # Check cache first
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._embedding_cache:
                    embedding = self._embedding_cache[cache_key]
                    result = EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        model_name=self.model_name,
                        processing_time_ms=0  # Cached
                    )
                    cached_results.append((i, result))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    
            # Generate embeddings for uncached texts
            new_results = []
            if uncached_texts:
                await self._load_model()
                
                # Process in batches
                batch_size = self.config.batch_size
                for i in range(0, len(uncached_texts), batch_size):
                    batch_texts = uncached_texts[i:i + batch_size]
                    batch_indices = uncached_indices[i:i + batch_size]
                    
                    # Generate embeddings in thread pool
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        None,
                        self._encode_batch,
                        batch_texts
                    )
                    
                    # Create results and cache
                    for j, (text, embedding) in enumerate(zip(batch_texts, embeddings)):
                        original_index = batch_indices[j]
                        
                        result = EmbeddingResult(
                            text=text,
                            embedding=embedding,
                            model_name=self.model_name,
                            processing_time_ms=int((time.time() - start_time) * 1000)
                        )
                        
                        new_results.append((original_index, result))
                        
                        # Cache the embedding
                        cache_key = self._get_cache_key(text)
                        self._cache_embedding(cache_key, embedding)
                        
            # Combine cached and new results in original order
            all_results = cached_results + new_results
            all_results.sort(key=lambda x: x[0])  # Sort by original index
            
            final_results = [result for _, result in all_results]
            
            self.logger.debug(
                f"Generated {len(final_results)} embeddings "
                f"({len(cached_results)} cached, {len(new_results)} new)"
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise
            
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Synchronously encode a batch of texts."""
        if not self.model:
            raise RuntimeError("Model not loaded")
            
        # Truncate texts to model's max sequence length if needed
        processed_texts = []
        for text in texts:
            # Simple truncation based on approximate token count
            max_chars = self.config.max_sequence_length * 4  # Rough estimate
            if len(text) > max_chars:
                text = text[:max_chars]
            processed_texts.append(text)
            
        # Generate embeddings
        embeddings = self.model.encode(
            processed_texts,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings
        
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Use hash of text + model name for cache key
        content = f"{self.model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
        
    def _cache_embedding(self, cache_key: str, embedding: np.ndarray) -> None:
        """Cache an embedding with LRU eviction."""
        # Simple LRU: remove oldest if at limit
        if len(self._embedding_cache) >= self._cache_size_limit:
            # Remove the first (oldest) item
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
            
        self._embedding_cache[cache_key] = embedding
        
    async def embed_and_store_chunk(
        self, 
        chunk_id: str, 
        text: str
    ) -> ChunkEmbedding:
        """
        Generate and store embedding for a document chunk.
        
        Args:
            chunk_id: Document chunk ID
            text: Text content to embed
            
        Returns:
            Stored chunk embedding
        """
        try:
            # Generate embedding
            result = await self.embed_text(text)
            
            # Serialize embedding
            embedding_bytes = pickle.dumps(result.embedding)
            
            # Create database record
            chunk_embedding = ChunkEmbedding(
                chunk_id=chunk_id,
                model_name=self.model_name,
                embedding_vector=embedding_bytes,
                vector_dimension=self.config.dimension
            )
            
            # Store in database
            async with self.database_manager.get_session() as session:
                session.add(chunk_embedding)
                await session.commit()
                await session.refresh(chunk_embedding)
                
            self.logger.debug(f"Stored embedding for chunk: {chunk_id}")
            return chunk_embedding
            
        except Exception as e:
            self.logger.error(f"Failed to embed and store chunk {chunk_id}: {e}")
            raise
            
    async def embed_and_store_conversation(
        self,
        conversation_id: str,
        content_summary: str,
        message_id: Optional[str] = None
    ) -> ConversationEmbedding:
        """
        Generate and store embedding for conversation content.
        
        Args:
            conversation_id: Conversation ID
            content_summary: Summarized content to embed
            message_id: Optional specific message ID
            
        Returns:
            Stored conversation embedding
        """
        try:
            # Generate embedding
            result = await self.embed_text(content_summary)
            
            # Serialize embedding
            embedding_bytes = pickle.dumps(result.embedding)
            
            # Create database record
            conv_embedding = ConversationEmbedding(
                conversation_id=conversation_id,
                message_id=message_id,
                content_summary=content_summary,
                model_name=self.model_name,
                embedding_vector=embedding_bytes,
                vector_dimension=self.config.dimension
            )
            
            # Store in database
            async with self.database_manager.get_session() as session:
                session.add(conv_embedding)
                await session.commit()
                await session.refresh(conv_embedding)
                
            self.logger.debug(f"Stored conversation embedding: {conversation_id}")
            return conv_embedding
            
        except Exception as e:
            self.logger.error(f"Failed to embed and store conversation {conversation_id}: {e}")
            raise
            
    async def get_similarity_scores(
        self, 
        query_embedding: np.ndarray,
        target_embeddings: List[np.ndarray]
    ) -> List[float]:
        """
        Calculate cosine similarity scores between query and target embeddings.
        
        Args:
            query_embedding: Query vector
            target_embeddings: List of target vectors
            
        Returns:
            List of similarity scores (0-1)
        """
        try:
            if not target_embeddings:
                return []
                
            # Stack target embeddings
            targets = np.stack(target_embeddings)
            
            # Calculate cosine similarity
            # Normalize vectors if not already normalized
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            targets_norm = targets / np.linalg.norm(targets, axis=1, keepdims=True)
            
            # Compute similarities
            similarities = np.dot(targets_norm, query_norm)
            
            # Convert to list and ensure values are in [0, 1]
            scores = np.clip(similarities, 0, 1).tolist()
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate similarity scores: {e}")
            raise
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            "model_name": self.model_name,
            "dimension": self.config.dimension,
            "max_sequence_length": self.config.max_sequence_length,
            "batch_size": self.config.batch_size,
            "device": self.config.device,
            "normalize_embeddings": self.config.normalize_embeddings,
            "cache_size": len(self._embedding_cache),
            "available_models": list(self.AVAILABLE_MODELS.keys())
        }
        
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available embedding models."""
        return list(cls.AVAILABLE_MODELS.keys())
        
    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[EmbeddingConfig]:
        """Get configuration for a specific model."""
        return cls.AVAILABLE_MODELS.get(model_name)