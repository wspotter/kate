"""
RAG configuration service for Kate LLM Client.

Manages comprehensive configuration settings for all RAG components including
chunk sizes, retrieval parameters, similarity thresholds, and performance tuning.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

from loguru import logger

from ..core.events import EventBus
from ..core.config import AppSettings


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BOUNDARY = "sentence_boundary"
    PARAGRAPH_BOUNDARY = "paragraph_boundary"
    SEMANTIC_BOUNDARY = "semantic_boundary"


class EmbeddingModel(Enum):
    """Available embedding models."""
    MINILM_L6_V2 = "all-MiniLM-L6-v2"
    MPNET_BASE_V2 = "all-mpnet-base-v2"
    DISTILBERT_BASE = "all-distilroberta-v1"
    BGE_SMALL = "BAAI/bge-small-en"
    BGE_BASE = "BAAI/bge-base-en"


@dataclass
class DocumentProcessingConfig:
    """Configuration for document processing."""
    # Chunking settings
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 64  # tokens
    min_chunk_size: int = 50  # tokens
    max_chunk_size: int = 2048  # tokens
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_BOUNDARY
    
    # Content processing
    strip_whitespace: bool = True
    normalize_unicode: bool = True
    remove_extra_newlines: bool = True
    preserve_formatting: bool = False
    
    # OCR settings
    enable_ocr: bool = True
    ocr_language: str = "eng"
    ocr_confidence_threshold: float = 0.7
    
    # Supported formats
    supported_formats: List[str] = field(default_factory=lambda: [
        "pdf", "docx", "txt", "html", "csv", "json", "xlsx", "xls", "md"
    ])


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    # Model settings
    model_name: EmbeddingModel = EmbeddingModel.MINILM_L6_V2
    device: str = "cpu"  # cpu, cuda, mps
    batch_size: int = 32
    max_sequence_length: int = 512
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 10000
    cache_ttl_hours: int = 24
    
    # Normalization
    normalize_embeddings: bool = True
    pooling_strategy: str = "mean"  # mean, cls, max


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations."""
    # Search parameters
    max_results: int = 10
    min_similarity_score: float = 0.7
    similarity_metric: str = "cosine"  # cosine, euclidean, dot_product
    
    # Ranking and filtering
    enable_reranking: bool = True
    diversity_threshold: float = 0.8
    temporal_decay_factor: float = 0.1
    popularity_boost: float = 0.05
    
    # Context window
    context_window_tokens: int = 4000
    enable_query_expansion: bool = True
    expansion_terms: int = 3
    
    # Conversation context
    conversation_context_messages: int = 5
    enable_conversation_weighting: bool = True
    conversation_weight: float = 1.2


@dataclass
class VectorStoreConfig:
    """Configuration for vector storage."""
    # ChromaDB settings
    collection_name_documents: str = "documents"
    collection_name_conversations: str = "conversations"
    distance_function: str = "cosine"
    
    # Index settings
    enable_hnsw: bool = True
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100
    
    # Storage settings
    persist_directory: str = "data/vector_store"
    backup_interval_hours: int = 24
    cleanup_threshold_days: int = 30


@dataclass
class RAGGenerationConfig:
    """Configuration for RAG response generation."""
    # Context integration
    max_context_tokens: int = 2000
    context_integration_strategy: str = "structured"  # prepend, structured, inline
    include_citations: bool = True
    citation_format: str = "[{source}]"
    
    # Generation parameters
    temperature: float = 0.7
    max_response_tokens: int = 1500
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Quality control
    enable_response_validation: bool = True
    enable_fact_checking: bool = False
    response_confidence_threshold: float = 0.6
    
    # Streaming
    enable_streaming: bool = True
    stream_chunk_size: int = 50


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    # Concurrency
    max_concurrent_embeddings: int = 4
    max_concurrent_retrievals: int = 3
    max_concurrent_generations: int = 2
    
    # Timeouts
    embedding_timeout_seconds: int = 30
    retrieval_timeout_seconds: int = 15
    generation_timeout_seconds: int = 60
    
    # Memory management
    embedding_cache_size_mb: int = 500
    vector_store_cache_size_mb: int = 1000
    enable_memory_monitoring: bool = True
    memory_cleanup_threshold: float = 0.85
    
    # Batching
    enable_batch_processing: bool = True
    batch_size_documents: int = 10
    batch_size_conversations: int = 5
    batch_timeout_seconds: int = 5


@dataclass
class RAGConfig:
    """Complete RAG system configuration."""
    # Component configurations
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    generation: RAGGenerationConfig = field(default_factory=RAGGenerationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Global settings
    enabled: bool = True
    debug_mode: bool = False
    version: str = "1.0.0"
    last_updated: Optional[str] = None


class RAGConfigService:
    """
    Service for managing RAG system configuration.
    
    Provides centralized configuration management with validation,
    persistence, and real-time updates.
    """
    
    def __init__(self, app_settings: AppSettings, event_bus: EventBus):
        self.app_settings = app_settings
        self.event_bus = event_bus
        self.logger = logger.bind(component="RAGConfigService")
        
        # Configuration state
        self._config: RAGConfig = RAGConfig()
        self._config_file_path: Optional[Path] = None
        self._validation_rules: Dict[str, Callable] = {}
        self._change_listeners: List[Callable] = []
        
        # Default presets
        self._presets = self._create_default_presets()
        
        self._setup_validation_rules()
        
    async def initialize(self) -> None:
        """Initialize the configuration service."""
        # Set up config file path
        config_dir = Path(self.app_settings.user_data_dir) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        self._config_file_path = config_dir / "rag_config.json"
        
        # Load existing configuration
        await self.load_config()
        
        self.logger.info("RAG configuration service initialized")
        
    async def cleanup(self) -> None:
        """Cleanup configuration service."""
        # Save current configuration
        await self.save_config()
        self.logger.info("RAG configuration service cleaned up")
        
    def get_config(self) -> RAGConfig:
        """Get the current RAG configuration."""
        return self._config
        
    def get_component_config(self, component: str) -> Any:
        """Get configuration for a specific component."""
        component_configs = {
            "document_processing": self._config.document_processing,
            "embedding": self._config.embedding,
            "retrieval": self._config.retrieval,
            "vector_store": self._config.vector_store,
            "generation": self._config.generation,
            "performance": self._config.performance
        }
        
        return component_configs.get(component)
        
    async def update_config(
        self, 
        updates: Dict[str, Any], 
        validate: bool = True,
        persist: bool = True
    ) -> bool:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            validate: Whether to validate the updates
            persist: Whether to persist changes to disk
            
        Returns:
            True if update was successful
        """
        try:
            # Create a copy for validation
            new_config = RAGConfig(**asdict(self._config))
            
            # Apply updates
            self._apply_updates(new_config, updates)
            
            # Validate if requested
            if validate:
                validation_errors = self._validate_config(new_config)
                if validation_errors:
                    self.logger.error(f"Configuration validation failed: {validation_errors}")
                    return False
                    
            # Update timestamp
            from datetime import datetime
            new_config.last_updated = datetime.now().isoformat()
            
            # Apply the changes
            old_config = self._config
            self._config = new_config
            
            # Persist if requested
            if persist:
                await self.save_config()
                
            # Notify listeners
            await self._notify_config_changed(old_config, new_config)
            
            self.logger.info("RAG configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
            
    def _apply_updates(self, config: RAGConfig, updates: Dict[str, Any]) -> None:
        """Apply configuration updates recursively."""
        for key, value in updates.items():
            if "." in key:
                # Nested key like "retrieval.max_results"
                parts = key.split(".", 1)
                component_name, sub_key = parts
                
                component = getattr(config, component_name, None)
                if component:
                    if "." in sub_key:
                        # Further nesting
                        self._apply_updates(component, {sub_key: value})
                    else:
                        setattr(component, sub_key, value)
            else:
                # Direct attribute
                if hasattr(config, key):
                    setattr(config, key, value)
                    
    def _validate_config(self, config: RAGConfig) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Document processing validation
        doc_config = config.document_processing
        if doc_config.chunk_size < doc_config.min_chunk_size:
            errors.append("chunk_size cannot be less than min_chunk_size")
        if doc_config.chunk_size > doc_config.max_chunk_size:
            errors.append("chunk_size cannot be greater than max_chunk_size")
        if doc_config.chunk_overlap >= doc_config.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")
            
        # Retrieval validation
        ret_config = config.retrieval
        if not (0.0 <= ret_config.min_similarity_score <= 1.0):
            errors.append("min_similarity_score must be between 0.0 and 1.0")
        if ret_config.max_results <= 0:
            errors.append("max_results must be positive")
        if ret_config.context_window_tokens <= 0:
            errors.append("context_window_tokens must be positive")
            
        # Embedding validation
        emb_config = config.embedding
        if emb_config.batch_size <= 0:
            errors.append("embedding batch_size must be positive")
        if emb_config.max_sequence_length <= 0:
            errors.append("max_sequence_length must be positive")
            
        # Generation validation
        gen_config = config.generation
        if not (0.0 <= gen_config.temperature <= 2.0):
            errors.append("temperature must be between 0.0 and 2.0")
        if gen_config.max_response_tokens <= 0:
            errors.append("max_response_tokens must be positive")
            
        # Performance validation
        perf_config = config.performance
        if perf_config.max_concurrent_embeddings <= 0:
            errors.append("max_concurrent_embeddings must be positive")
        if perf_config.embedding_timeout_seconds <= 0:
            errors.append("embedding_timeout_seconds must be positive")
            
        return errors
        
    async def load_config(self) -> bool:
        """Load configuration from file."""
        if not self._config_file_path or not self._config_file_path.exists():
            self.logger.info("No existing configuration file, using defaults")
            return True
            
        try:
            with open(self._config_file_path, 'r') as f:
                config_dict = json.load(f)
                
            # Reconstruct config object
            self._config = self._dict_to_config(config_dict)
            
            # Validate loaded config
            validation_errors = self._validate_config(self._config)
            if validation_errors:
                self.logger.warning(f"Loaded configuration has validation errors: {validation_errors}")
                # Use defaults for invalid parts
                self._config = RAGConfig()
                
            self.logger.info("Configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self._config = RAGConfig()  # Fall back to defaults
            return False
            
    async def save_config(self) -> bool:
        """Save configuration to file."""
        if not self._config_file_path:
            return False
            
        try:
            config_dict = self._config_to_dict(self._config)
            
            # Ensure directory exists
            self._config_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write with atomic operation
            temp_path = self._config_file_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
                
            # Atomic rename
            temp_path.replace(self._config_file_path)
            
            self.logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
            
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> RAGConfig:
        """Convert dictionary to RAGConfig object."""
        # Convert enum strings back to enum values
        if "document_processing" in config_dict:
            doc_proc = config_dict["document_processing"]
            if "chunking_strategy" in doc_proc:
                doc_proc["chunking_strategy"] = ChunkingStrategy(doc_proc["chunking_strategy"])
                
        if "embedding" in config_dict:
            embedding = config_dict["embedding"]
            if "model_name" in embedding:
                embedding["model_name"] = EmbeddingModel(embedding["model_name"])
                
        # Create config object
        return RAGConfig(
            document_processing=DocumentProcessingConfig(**config_dict.get("document_processing", {})),
            embedding=EmbeddingConfig(**config_dict.get("embedding", {})),
            retrieval=RetrievalConfig(**config_dict.get("retrieval", {})),
            vector_store=VectorStoreConfig(**config_dict.get("vector_store", {})),
            generation=RAGGenerationConfig(**config_dict.get("generation", {})),
            performance=PerformanceConfig(**config_dict.get("performance", {})),
            enabled=config_dict.get("enabled", True),
            debug_mode=config_dict.get("debug_mode", False),
            version=config_dict.get("version", "1.0.0"),
            last_updated=config_dict.get("last_updated")
        )
        
    def _config_to_dict(self, config: RAGConfig) -> Dict[str, Any]:
        """Convert RAGConfig object to dictionary."""
        config_dict = asdict(config)
        
        # Convert enums to strings for JSON serialization
        if "document_processing" in config_dict:
            doc_proc = config_dict["document_processing"]
            if "chunking_strategy" in doc_proc:
                doc_proc["chunking_strategy"] = doc_proc["chunking_strategy"].value
                
        if "embedding" in config_dict:
            embedding = config_dict["embedding"]
            if "model_name" in embedding:
                embedding["model_name"] = embedding["model_name"].value
                
        return config_dict
        
    def _setup_validation_rules(self) -> None:
        """Set up validation rules for configuration."""
        self._validation_rules = {
            "chunk_size": lambda x: isinstance(x, int) and 50 <= x <= 4096,
            "min_similarity_score": lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
            "max_results": lambda x: isinstance(x, int) and 1 <= x <= 100,
            "temperature": lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 2.0,
            "batch_size": lambda x: isinstance(x, int) and 1 <= x <= 128,
        }
        
    def _create_default_presets(self) -> Dict[str, RAGConfig]:
        """Create default configuration presets."""
        # Performance preset
        performance_config = RAGConfig()
        performance_config.document_processing.chunk_size = 256
        performance_config.retrieval.max_results = 5
        performance_config.embedding.batch_size = 64
        performance_config.performance.max_concurrent_embeddings = 8
        
        # Quality preset
        quality_config = RAGConfig()
        quality_config.document_processing.chunk_size = 1024
        quality_config.retrieval.max_results = 15
        quality_config.retrieval.min_similarity_score = 0.8
        quality_config.embedding.model_name = EmbeddingModel.MPNET_BASE_V2
        quality_config.generation.enable_fact_checking = True
        
        # Balanced preset (default)
        balanced_config = RAGConfig()
        
        return {
            "performance": performance_config,
            "quality": quality_config,
            "balanced": balanced_config
        }
        
    def get_presets(self) -> Dict[str, str]:
        """Get available configuration presets."""
        return {
            "performance": "Optimized for speed and efficiency",
            "quality": "Optimized for accuracy and relevance",
            "balanced": "Balanced performance and quality"
        }
        
    async def apply_preset(self, preset_name: str) -> bool:
        """Apply a configuration preset."""
        if preset_name not in self._presets:
            self.logger.error(f"Unknown preset: {preset_name}")
            return False
            
        preset_config = self._presets[preset_name]
        self._config = RAGConfig(**asdict(preset_config))
        
        await self.save_config()
        await self._notify_config_changed(None, self._config)
        
        self.logger.info(f"Applied configuration preset: {preset_name}")
        return True
        
    def add_change_listener(self, listener: Callable) -> None:
        """Add a configuration change listener."""
        self._change_listeners.append(listener)
        
    def remove_change_listener(self, listener: Callable) -> None:
        """Remove a configuration change listener."""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
            
    async def _notify_config_changed(self, old_config: Optional[RAGConfig], new_config: RAGConfig) -> None:
        """Notify listeners of configuration changes."""
        # Notify event bus
        await self.event_bus.emit("rag_config_changed", {
            "old_config": asdict(old_config) if old_config else None,
            "new_config": asdict(new_config),
            "timestamp": new_config.last_updated
        })
        
        # Notify direct listeners
        for listener in self._change_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(old_config, new_config)
                else:
                    listener(old_config, new_config)
            except Exception as e:
                self.logger.warning(f"Configuration change listener failed: {e}")
                
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration as dictionary."""
        return self._config_to_dict(self._config)
        
    async def import_config(self, config_dict: Dict[str, Any], validate: bool = True) -> bool:
        """Import configuration from dictionary."""
        try:
            new_config = self._dict_to_config(config_dict)
            
            if validate:
                validation_errors = self._validate_config(new_config)
                if validation_errors:
                    self.logger.error(f"Imported configuration is invalid: {validation_errors}")
                    return False
                    
            old_config = self._config
            self._config = new_config
            
            await self.save_config()
            await self._notify_config_changed(old_config, new_config)
            
            self.logger.info("Configuration imported successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return False
            
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            "enabled": self._config.enabled,
            "chunk_size": self._config.document_processing.chunk_size,
            "embedding_model": self._config.embedding.model_name.value,
            "max_results": self._config.retrieval.max_results,
            "min_similarity": self._config.retrieval.min_similarity_score,
            "context_window": self._config.retrieval.context_window_tokens,
            "temperature": self._config.generation.temperature,
            "max_response_tokens": self._config.generation.max_response_tokens,
            "last_updated": self._config.last_updated,
            "version": self._config.version
        }
        
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._config = RAGConfig()
        self.logger.info("Configuration reset to defaults")