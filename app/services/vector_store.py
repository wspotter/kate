"""
Vector store service for Kate LLM Client using ChromaDB.

This service provides efficient similarity search capabilities for RAG
by integrating ChromaDB with the embedding service and database models.
"""

import asyncio
import pickle
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass

import numpy as np
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

from ..core.events import EventBus
from ..database.manager import DatabaseManager
from ..database.models import DocumentChunk, ChunkEmbedding, ConversationEmbedding
from .embedding_service import EmbeddingService


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    document_title: Optional[str] = None
    document_id: Optional[str] = None


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    persist_directory: Path
    collection_name: str = "kate_documents"
    conversation_collection_name: str = "kate_conversations"
    distance_metric: str = "cosine"  # cosine, euclidean, ip
    max_results: int = 10
    score_threshold: float = 0.7


class VectorStore:
    """
    Vector store service using ChromaDB for efficient similarity search.
    
    Manages document and conversation embeddings for RAG retrieval,
    providing fast semantic search capabilities.
    """
    
    def __init__(
        self,
        database_manager: DatabaseManager,
        embedding_service: EmbeddingService,
        event_bus: EventBus,
        config: Optional[VectorStoreConfig] = None
    ):
        if not HAS_CHROMADB:
            raise RuntimeError("ChromaDB not available. Install with: pip install chromadb")
            
        self.database_manager = database_manager
        self.embedding_service = embedding_service
        self.event_bus = event_bus
        
        # Configuration
        default_persist_dir = Path.home() / ".cache" / "kate" / "vector_store"
        self.config = config or VectorStoreConfig(persist_directory=default_persist_dir)
        
        # Ensure persist directory exists
        self.config.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB client and collections
        self.client: Optional[chromadb.PersistentClient] = None
        self.document_collection = None
        self.conversation_collection = None
        
        # Logger
        self.logger = logger.bind(component="VectorStore")
        
    async def initialize(self) -> None:
        """Initialize the vector store and create collections."""
        try:
            self.logger.info("Initializing ChromaDB vector store")
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.config.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get document collection
            self.document_collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={
                    "description": "Document chunks for RAG",
                    "distance_metric": self.config.distance_metric
                }
            )
            
            # Create or get conversation collection
            self.conversation_collection = self.client.get_or_create_collection(
                name=self.config.conversation_collection_name,
                metadata={
                    "description": "Conversation embeddings for context retrieval",
                    "distance_metric": self.config.distance_metric
                }
            )
            
            self.logger.info("Vector store initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            raise
            
    async def cleanup(self) -> None:
        """Cleanup vector store resources."""
        try:
            # ChromaDB client cleanup is automatic
            self.client = None
            self.document_collection = None
            self.conversation_collection = None
            
            self.logger.info("Vector store cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    async def add_document_chunk(
        self, 
        chunk_id: str, 
        content: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a document chunk to the vector store.
        
        Args:
            chunk_id: Unique identifier for the chunk
            content: Text content of the chunk
            embedding: Pre-computed embedding (will generate if None)
            metadata: Additional metadata for the chunk
        """
        try:
            if not self.document_collection:
                raise RuntimeError("Vector store not initialized")
                
            # Generate embedding if not provided
            if embedding is None:
                embedding_result = await self.embedding_service.embed_text(content)
                embedding = embedding_result.embedding
                
            # Prepare metadata
            chunk_metadata = metadata or {}
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "content_length": len(content),
                "word_count": len(content.split()),
                "embedding_model": self.embedding_service.model_name
            })
            
            # Add to ChromaDB
            self.document_collection.add(
                ids=[chunk_id],
                embeddings=[embedding.tolist()],
                documents=[content],
                metadatas=[chunk_metadata]
            )
            
            self.logger.debug(f"Added document chunk to vector store: {chunk_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add document chunk {chunk_id}: {e}")
            raise
            
    async def add_conversation_embedding(
        self,
        conversation_id: str,
        content_summary: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a conversation embedding to the vector store.
        
        Args:
            conversation_id: Conversation identifier
            content_summary: Summarized conversation content
            embedding: Pre-computed embedding (will generate if None)
            metadata: Additional metadata
        """
        try:
            if not self.conversation_collection:
                raise RuntimeError("Vector store not initialized")
                
            # Generate embedding if not provided
            if embedding is None:
                embedding_result = await self.embedding_service.embed_text(content_summary)
                embedding = embedding_result.embedding
                
            # Create unique ID for this embedding
            embedding_id = f"conv_{conversation_id}_{uuid.uuid4().hex[:8]}"
            
            # Prepare metadata
            conv_metadata = metadata or {}
            conv_metadata.update({
                "conversation_id": conversation_id,
                "content_length": len(content_summary),
                "word_count": len(content_summary.split()),
                "embedding_model": self.embedding_service.model_name
            })
            
            # Add to ChromaDB
            self.conversation_collection.add(
                ids=[embedding_id],
                embeddings=[embedding.tolist()],
                documents=[content_summary],
                metadatas=[conv_metadata]
            )
            
            self.logger.debug(f"Added conversation embedding to vector store: {conversation_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add conversation embedding {conversation_id}: {e}")
            raise
            
    async def search_documents(
        self,
        query: str,
        max_results: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar document chunks.
        
        Args:
            query: Search query text
            max_results: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_metadata: Metadata filters to apply
            
        Returns:
            List of search results
        """
        try:
            if not self.document_collection:
                raise RuntimeError("Vector store not initialized")
                
            # Generate query embedding
            embedding_result = await self.embedding_service.embed_text(query)
            query_embedding = embedding_result.embedding
            
            # Set defaults
            max_results = max_results or self.config.max_results
            score_threshold = score_threshold or self.config.score_threshold
            
            # Perform search
            results = self.document_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=max_results,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to SearchResult objects
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    # Convert distance to similarity score (assuming cosine distance)
                    distance = results["distances"][0][i]
                    score = 1.0 - distance  # Convert distance to similarity
                    
                    # Apply score threshold
                    if score < score_threshold:
                        continue
                        
                    content = results["documents"][0][i]
                    metadata = results["metadatas"][0][i] or {}
                    
                    search_result = SearchResult(
                        chunk_id=chunk_id,
                        content=content,
                        score=score,
                        metadata=metadata,
                        document_title=metadata.get("document_title"),
                        document_id=metadata.get("document_id")
                    )
                    
                    search_results.append(search_result)
                    
            self.logger.debug(f"Document search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Document search failed: {e}")
            raise
            
    async def search_conversations(
        self,
        query: str,
        max_results: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar conversation content.
        
        Args:
            query: Search query text
            max_results: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_metadata: Metadata filters to apply
            
        Returns:
            List of search results
        """
        try:
            if not self.conversation_collection:
                raise RuntimeError("Vector store not initialized")
                
            # Generate query embedding
            embedding_result = await self.embedding_service.embed_text(query)
            query_embedding = embedding_result.embedding
            
            # Set defaults
            max_results = max_results or self.config.max_results
            score_threshold = score_threshold or self.config.score_threshold
            
            # Perform search
            results = self.conversation_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=max_results,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to SearchResult objects
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, embedding_id in enumerate(results["ids"][0]):
                    # Convert distance to similarity score
                    distance = results["distances"][0][i]
                    score = 1.0 - distance
                    
                    # Apply score threshold
                    if score < score_threshold:
                        continue
                        
                    content = results["documents"][0][i]
                    metadata = results["metadatas"][0][i] or {}
                    
                    search_result = SearchResult(
                        chunk_id=embedding_id,
                        content=content,
                        score=score,
                        metadata=metadata,
                        document_id=metadata.get("conversation_id")
                    )
                    
                    search_results.append(search_result)
                    
            self.logger.debug(f"Conversation search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Conversation search failed: {e}")
            raise
            
    async def batch_add_chunks(self, chunks_data: List[Tuple[str, str, Dict[str, Any]]]) -> None:
        """
        Batch add multiple document chunks for efficiency.
        
        Args:
            chunks_data: List of (chunk_id, content, metadata) tuples
        """
        try:
            if not self.document_collection or not chunks_data:
                return
                
            # Extract data for batch processing
            chunk_ids = []
            contents = []
            metadatas = []
            
            for chunk_id, content, metadata in chunks_data:
                chunk_ids.append(chunk_id)
                contents.append(content)
                
                # Prepare metadata
                chunk_metadata = metadata or {}
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "content_length": len(content),
                    "word_count": len(content.split()),
                    "embedding_model": self.embedding_service.model_name
                })
                metadatas.append(chunk_metadata)
                
            # Generate embeddings in batch
            embedding_results = await self.embedding_service.embed_texts(contents)
            embeddings = [result.embedding.tolist() for result in embedding_results]
            
            # Add to ChromaDB in batch
            self.document_collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )
            
            self.logger.info(f"Batch added {len(chunks_data)} chunks to vector store")
            
        except Exception as e:
            self.logger.error(f"Batch add chunks failed: {e}")
            raise
            
    async def delete_document_chunks(self, chunk_ids: List[str]) -> None:
        """
        Delete document chunks from the vector store.
        
        Args:
            chunk_ids: List of chunk IDs to delete
        """
        try:
            if not self.document_collection or not chunk_ids:
                return
                
            self.document_collection.delete(ids=chunk_ids)
            self.logger.debug(f"Deleted {len(chunk_ids)} chunks from vector store")
            
        except Exception as e:
            self.logger.error(f"Failed to delete chunks: {e}")
            raise
            
    async def delete_conversation_embeddings(self, conversation_id: str) -> None:
        """
        Delete all embeddings for a conversation.
        
        Args:
            conversation_id: Conversation ID to delete embeddings for
        """
        try:
            if not self.conversation_collection:
                return
                
            # Query for embeddings with this conversation_id
            results = self.conversation_collection.get(
                where={"conversation_id": conversation_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                self.conversation_collection.delete(ids=results["ids"])
                self.logger.debug(f"Deleted conversation embeddings: {conversation_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to delete conversation embeddings: {e}")
            raise
            
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collections.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = {
                "initialized": self.client is not None,
                "document_collection": {
                    "name": self.config.collection_name,
                    "count": 0,
                    "metadata": {}
                },
                "conversation_collection": {
                    "name": self.config.conversation_collection_name,
                    "count": 0,
                    "metadata": {}
                }
            }
            
            if self.document_collection:
                doc_count = self.document_collection.count()
                stats["document_collection"]["count"] = doc_count
                stats["document_collection"]["metadata"] = self.document_collection.metadata
                
            if self.conversation_collection:
                conv_count = self.conversation_collection.count()
                stats["conversation_collection"]["count"] = conv_count
                stats["conversation_collection"]["metadata"] = self.conversation_collection.metadata
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
            
    async def rebuild_index(self) -> None:
        """
        Rebuild the vector store index from database records.
        
        This is useful for recovering from vector store corruption
        or migrating to a new embedding model.
        """
        try:
            self.logger.info("Starting vector store index rebuild")
            
            # Clear existing collections
            if self.document_collection:
                self.document_collection.delete()
                
            if self.conversation_collection:
                self.conversation_collection.delete()
                
            # Recreate collections
            await self.initialize()
            
            # Rebuild document chunks
            async with self.database_manager.get_session() as session:
                # Get all chunk embeddings
                chunk_embeddings = await session.execute(
                    "SELECT chunk_id, embedding_vector FROM chunk_embeddings"
                )
                
                chunks_to_add = []
                for row in chunk_embeddings:
                    chunk_id = row.chunk_id
                    embedding_bytes = row.embedding_vector
                    
                    # Deserialize embedding
                    embedding = pickle.loads(embedding_bytes)
                    
                    # Get chunk content
                    chunk = await session.get(DocumentChunk, chunk_id)
                    if chunk:
                        chunks_to_add.append((chunk_id, chunk.content, {
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.chunk_index
                        }))
                        
                # Batch add chunks
                if chunks_to_add:
                    await self.batch_add_chunks(chunks_to_add)
                    
                # Rebuild conversation embeddings
                conv_embeddings = await session.execute(
                    "SELECT id, conversation_id, content_summary, embedding_vector FROM conversation_embeddings"
                )
                
                for row in conv_embeddings:
                    embedding_id = row.id
                    conversation_id = row.conversation_id
                    content = row.content_summary
                    embedding_bytes = row.embedding_vector
                    
                    # Deserialize embedding
                    embedding = pickle.loads(embedding_bytes)
                    
                    # Add to conversation collection
                    await self.add_conversation_embedding(
                        conversation_id=conversation_id,
                        content_summary=content,
                        embedding=embedding,
                        metadata={"embedding_id": embedding_id}
                    )
                    
            self.logger.info("Vector store index rebuild completed")
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild index: {e}")
            raise
            
    def get_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        return {
            "persist_directory": str(self.config.persist_directory),
            "collection_name": self.config.collection_name,
            "conversation_collection_name": self.config.conversation_collection_name,
            "distance_metric": self.config.distance_metric,
            "max_results": self.config.max_results,
            "score_threshold": self.config.score_threshold,
            "has_chromadb": HAS_CHROMADB
        }