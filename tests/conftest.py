"""
Test configuration and fixtures for Kate LLM Client RAG system tests.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from app.core.config import AppSettings
from app.core.events import EventBus

# Import the modules we'll be testing
from app.database.manager import DatabaseManager
from app.database.models import Conversation, Document, DocumentChunk, Message
from app.services.background_processing_service import BackgroundProcessingService
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.rag_evaluation_service import RAGEvaluationService, RetrievalContext
from app.services.rag_integration_service import RAGIntegrationService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import VectorStore


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def app_settings(temp_dir):
    """Create test app settings."""
    return AppSettings(
        database_url=f"sqlite:///{temp_dir}/test.db",
        vector_store_path=str(temp_dir / "vector_store"),
        document_store_path=str(temp_dir / "documents"),
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=50,
        max_retrieval_docs=5,
        similarity_threshold=0.7
    )


@pytest.fixture
async def database_manager(app_settings):
    """Create test database manager."""
    db_manager = DatabaseManager(app_settings.database_url)
    await db_manager.initialize()
    yield db_manager
    await db_manager.close()


@pytest.fixture
def event_bus():
    """Create test event bus."""
    return EventBus()


@pytest.fixture
def mock_embedding_service():
    """Create mock embedding service."""
    service = Mock(spec=EmbeddingService)
    service.embed_text = AsyncMock(return_value=np.random.rand(384).tolist())
    service.embed_texts = AsyncMock(return_value=[np.random.rand(384).tolist() for _ in range(3)])
    service.get_embedding_dimension = Mock(return_value=384)
    return service


@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    provider = Mock()
    provider.generate_response = AsyncMock(return_value="This is a test response.")
    provider.generate_streaming_response = AsyncMock()
    return provider


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "title": "Test Document",
        "filename": "test_doc.txt",
        "content": "This is a sample document for testing purposes. It contains multiple sentences to test chunking and retrieval.",
        "file_type": "text/plain",
        "file_size": 1024,
        "metadata": {"author": "Test Author", "category": "testing"}
    }


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "content": "This is the first chunk of the document.",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 40,
            "metadata": {"section": "intro"}
        },
        {
            "content": "This is the second chunk with more content.",
            "chunk_index": 1,
            "start_char": 41,
            "end_char": 85,
            "metadata": {"section": "body"}
        },
        {
            "content": "This is the final chunk of the test document.",
            "chunk_index": 2,
            "start_char": 86,
            "end_char": 132,
            "metadata": {"section": "conclusion"}
        }
    ]


@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing."""
    return {
        "title": "Test Conversation",
        "messages": [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence."},
            {"role": "user", "content": "Can you give me more details?"},
            {"role": "assistant", "content": "Machine learning uses algorithms to learn patterns from data."}
        ]
    }


@pytest.fixture
def sample_retrieval_context():
    """Sample retrieval context for testing."""
    return RetrievalContext(
        document_chunks=[],  # Will be populated by tests
        similarity_scores=[0.9, 0.8, 0.7],
        retrieval_query="test query",
        total_retrieved=3,
        retrieval_time=0.5
    )


@pytest.fixture
def sample_test_files(temp_dir):
    """Create sample test files."""
    files = {}
    
    # Text file
    txt_file = temp_dir / "sample.txt"
    txt_file.write_text("This is a sample text file for testing document processing.")
    files["txt"] = txt_file
    
    # JSON file
    json_file = temp_dir / "sample.json"
    json_file.write_text('{"title": "Sample JSON", "content": "This is JSON content"}')
    files["json"] = json_file
    
    # CSV content (simplified)
    csv_file = temp_dir / "sample.csv"
    csv_file.write_text("name,value\ntest1,100\ntest2,200\n")
    files["csv"] = csv_file
    
    return files


@pytest.fixture
async def document_processor(app_settings, mock_embedding_service):
    """Create test document processor."""
    processor = DocumentProcessor(app_settings, mock_embedding_service)
    return processor


@pytest.fixture
async def vector_store(app_settings):
    """Create test vector store."""
    store = VectorStore(app_settings.vector_store_path)
    await store.initialize()
    yield store
    await store.cleanup()


@pytest.fixture
async def retrieval_service(database_manager, mock_embedding_service, vector_store):
    """Create test retrieval service."""
    service = RetrievalService(database_manager, mock_embedding_service, vector_store)
    return service


@pytest.fixture
async def rag_integration_service(database_manager, mock_embedding_service, vector_store, mock_llm_provider):
    """Create test RAG integration service."""
    retrieval_service = RetrievalService(database_manager, mock_embedding_service, vector_store)
    service = RAGIntegrationService(
        database_manager=database_manager,
        embedding_service=mock_embedding_service,
        retrieval_service=retrieval_service,
        llm_providers={"test": mock_llm_provider}
    )
    return service


@pytest.fixture
def rag_evaluation_service(mock_embedding_service):
    """Create test RAG evaluation service."""
    service = RAGEvaluationService(mock_embedding_service)
    return service


@pytest.fixture
async def background_processing_service(document_processor, mock_embedding_service, vector_store):
    """Create test background processing service."""
    service = BackgroundProcessingService(document_processor, mock_embedding_service, vector_store)
    return service


# Test data generators
def generate_test_embedding(dimension: int = 384) -> List[float]:
    """Generate a test embedding vector."""
    return np.random.rand(dimension).tolist()


def generate_test_documents(count: int = 3) -> List[Dict[str, Any]]:
    """Generate test document data."""
    documents = []
    for i in range(count):
        documents.append({
            "title": f"Test Document {i+1}",
            "filename": f"test_doc_{i+1}.txt",
            "content": f"This is test document number {i+1}. It contains sample content for testing.",
            "file_type": "text/plain",
            "file_size": 1024 + i * 100,
            "metadata": {"index": i, "category": "test"}
        })
    return documents


# Async test helpers
@pytest.fixture
def async_test():
    """Helper for async test functions."""
    def _async_test(coro):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(coro)
    return _async_test


# Mock objects for UI testing
@pytest.fixture
def mock_qt_application():
    """Mock QApplication for UI tests."""
    from unittest.mock import Mock
    app = Mock()
    app.exec = Mock(return_value=0)
    app.quit = Mock()
    return app


@pytest.fixture
def mock_qt_widget():
    """Mock QWidget for UI tests."""
    from unittest.mock import Mock
    widget = Mock()
    widget.show = Mock()
    widget.hide = Mock()
    widget.setVisible = Mock()
    widget.isVisible = Mock(return_value=True)
    return widget


# Test configuration
@pytest.fixture(autouse=True)
def configure_logging():
    """Configure logging for tests."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # Reduce log noise during tests


# Performance test helpers
@pytest.fixture
def performance_timer():
    """Timer for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.time()
            
        def stop(self):
            self.end_time = time.time()
            
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
            
    return Timer()


# Integration test markers
pytestmark = [
    pytest.mark.asyncio,  # Most tests will be async
]