"""
SQLAlchemy 2.0 async database models for Kate LLM Client.
"""

import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class ContentType(enum.Enum):
    """Enumeration for different content types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    DOCUMENT = "document"
    MIXED = "mixed"  # For multi-modal content


class ProcessingStatus(enum.Enum):
    """Enumeration for content processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models."""
    pass


class Conversation(Base):
    """Model for storing conversations."""
    __tablename__ = "conversations"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """Model for storing individual messages in conversations."""
    __tablename__ = "messages"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversations.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    provider: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
    attachments: Mapped[List["FileAttachment"]] = relationship("FileAttachment", back_populates="message", cascade="all, delete-orphan")
    media_content: Mapped[List["MediaContent"]] = relationship("MediaContent", back_populates="message", cascade="all, delete-orphan")
    code_executions: Mapped[List["CodeExecution"]] = relationship("CodeExecution", back_populates="message", cascade="all, delete-orphan")


class Assistant(Base):
    """Model for storing assistant configurations and personas."""
    __tablename__ = "assistants"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    temperature: Mapped[float] = mapped_column(nullable=False, default=0.7)
    max_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class FileAttachment(Base):
    """(DEPRECATED) Model for storing file attachments to messages. Replaced by MediaContent."""
    __tablename__ = "file_attachments"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id: Mapped[str] = mapped_column(String(36), ForeignKey("messages.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    file_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    message: Mapped["Message"] = relationship("Message", back_populates="attachments")
    documents: Mapped[List["Document"]] = relationship("Document", back_populates="file_attachment", cascade="all, delete-orphan")


class Document(Base):
    """Model for storing processed documents in the knowledge base."""
    __tablename__ = "documents"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[ContentType] = mapped_column(Enum(ContentType), server_default=ContentType.DOCUMENT.value, nullable=False)
    file_attachment_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("file_attachments.id"), nullable=True) # Legacy
    media_content_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("media_content.id"), nullable=True)
    source_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    doc_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA-256 hash for deduplication
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    file_attachment: Mapped[Optional["FileAttachment"]] = relationship("FileAttachment", back_populates="documents")
    media_content: Mapped[Optional["MediaContent"]] = relationship("MediaContent")
    chunks: Mapped[List["DocumentChunk"]] = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Model for storing document chunks for RAG retrieval."""
    __tablename__ = "document_chunks"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)  # Order within document
    content: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    start_char: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Start position in original document
    end_char: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # End position in original document
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    embeddings: Mapped[List["ChunkEmbedding"]] = relationship("ChunkEmbedding", back_populates="chunk", cascade="all, delete-orphan")


class ChunkEmbedding(Base):
    """Model for storing vector embeddings of document chunks."""
    __tablename__ = "chunk_embeddings"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    chunk_id: Mapped[str] = mapped_column(String(36), ForeignKey("document_chunks.id"), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., 'all-MiniLM-L6-v2'
    embedding_vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)  # Serialized numpy array
    vector_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    chunk: Mapped["DocumentChunk"] = relationship("DocumentChunk", back_populates="embeddings")


class ConversationEmbedding(Base):
    """Model for storing vector embeddings of conversation content for context retrieval."""
    __tablename__ = "conversation_embeddings"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversations.id"), nullable=False)
    message_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("messages.id"), nullable=True)
    content_summary: Mapped[str] = mapped_column(Text, nullable=False)  # Summarized content for embedding
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    vector_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation")
    message: Mapped[Optional["Message"]] = relationship("Message")


class RAGSession(Base):
    """Model for tracking RAG retrieval sessions and context."""
    __tablename__ = "rag_sessions"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversations.id"), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunks: Mapped[List[str]] = mapped_column(JSON, nullable=False)  # List of chunk IDs
    context_used: Mapped[str] = mapped_column(Text, nullable=False)  # Combined context sent to LLM
    response_quality: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Quality score 0-1
    retrieval_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    generation_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation")


class MediaContent(Base):
    """Model for storing multi-modal media content (images, audio, video)."""
    __tablename__ = "media_content"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("messages.id"), nullable=True)
    content_type: Mapped[ContentType] = mapped_column(Enum(ContentType), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)  # Path to stored file
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Image-specific metadata
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Audio/Video-specific metadata
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    channels: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Processing status and metadata
    processing_status: Mapped[ProcessingStatus] = mapped_column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Extracted content and analysis
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # OCR, transcription, etc.
    analysis_results: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)  # Vision/audio analysis
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    message: Mapped[Optional["Message"]] = relationship("Message")
    embeddings: Mapped[List["MediaEmbedding"]] = relationship("MediaEmbedding", back_populates="media_content", cascade="all, delete-orphan")


class MediaEmbedding(Base):
    """Model for storing vector embeddings of multi-modal content."""
    __tablename__ = "media_embeddings"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    media_content_id: Mapped[str] = mapped_column(String(36), ForeignKey("media_content.id"), nullable=False)
    embedding_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'visual', 'audio', 'text', 'clip'
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., 'clip-vit-base-patch32'
    embedding_vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)  # Serialized numpy array
    vector_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    media_content: Mapped["MediaContent"] = relationship("MediaContent", back_populates="embeddings")


class CodeExecution(Base):
    """Model for storing code execution sessions and results."""
    __tablename__ = "code_executions"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("messages.id"), nullable=True)
    language: Mapped[str] = mapped_column(String(50), nullable=False)  # 'python', 'javascript', 'bash', etc.
    code: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Execution results
    status: Mapped[ProcessingStatus] = mapped_column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    stdout: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stderr: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    return_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Security and sandboxing
    sandbox_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    resource_limits: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    message: Mapped[Optional["Message"]] = relationship("Message")


class VisionAnalysis(Base):
    """Model for storing vision analysis results from multi-modal models."""
    __tablename__ = "vision_analyses"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    media_content_id: Mapped[str] = mapped_column(String(36), ForeignKey("media_content.id"), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., 'gpt-4-vision', 'gemini-pro-vision'
    
    # Analysis results
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # General description
    objects_detected: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)  # Object detection
    text_extracted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # OCR results
    labels: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)  # Classification labels
    confidence_scores: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON, nullable=True)
    
    # Structured analysis
    analysis_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'general', 'ocr', 'objects', 'scene'
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    media_content: Mapped["MediaContent"] = relationship("MediaContent")


class MultiModalSession(Base):
    """Model for tracking multi-modal AI sessions with combined content types."""
    __tablename__ = "multimodal_sessions"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversations.id"), nullable=False)
    session_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'vision_chat', 'voice_chat', 'code_analysis'
    
    # Content references
    text_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    media_content_ids: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)  # Referenced media IDs
    code_execution_ids: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)  # Referenced code executions
    
    # Model and provider info
    provider: Mapped[str] = mapped_column(String(50), nullable=False)  # 'openai', 'google', 'anthropic'
    model: Mapped[str] = mapped_column(String(100), nullable=False)  # 'gpt-4-vision', 'gemini-2.5-pro'
    model_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Session metrics
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost_estimate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation")