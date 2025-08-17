"""
Tests for database models and manager.
"""

import asyncio
from datetime import datetime
from typing import List

import pytest

from app.database.manager import DatabaseManager
from app.database.models import Conversation, Document, DocumentChunk, Embedding, Message


class TestDatabaseModels:
    """Test database model creation and relationships."""
    
    def test_document_creation(self):
        """Test document model creation."""
        document = Document(
            title="Test Document",
            filename="test.txt",
            file_type="text/plain",
            file_size=1024,
            content="Test content",
            metadata={"author": "Test Author"}
        )
        
        assert document.title == "Test Document"
        assert document.filename == "test.txt"
        assert document.file_type == "text/plain"
        assert document.file_size == 1024
        assert document.content == "Test content"
        assert document.metadata["author"] == "Test Author"
        assert document.processed is False
        assert isinstance(document.created_at, datetime)
        
    def test_document_chunk_creation(self):
        """Test document chunk model creation."""
        chunk = DocumentChunk(
            document_id="doc123",
            content="This is a chunk",
            chunk_index=0,
            start_char=0,
            end_char=15,
            metadata={"section": "intro"}
        )
        
        assert chunk.document_id == "doc123"
        assert chunk.content == "This is a chunk"
        assert chunk.chunk_index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 15
        assert chunk.metadata["section"] == "intro"
        
    def test_conversation_creation(self):
        """Test conversation model creation."""
        conversation = Conversation(
            title="Test Chat",
            metadata={"context": "testing"}
        )
        
        assert conversation.title == "Test Chat"
        assert conversation.metadata["context"] == "testing"
        assert isinstance(conversation.created_at, datetime)
        
    def test_message_creation(self):
        """Test message model creation."""
        message = Message(
            conversation_id="conv123",
            role="user",
            content="Hello, world!",
            metadata={"source": "test"}
        )
        
        assert message.conversation_id == "conv123"
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.metadata["source"] == "test"
        assert isinstance(message.timestamp, datetime)
        
    def test_embedding_creation(self):
        """Test embedding model creation."""
        embedding = Embedding(
            content_id="chunk123",
            content_type="document_chunk",
            embedding_vector=[0.1, 0.2, 0.3],
            model_name="test-model",
            metadata={"dimension": 3}
        )
        
        assert embedding.content_id == "chunk123"
        assert embedding.content_type == "document_chunk"
        assert embedding.embedding_vector == [0.1, 0.2, 0.3]
        assert embedding.model_name == "test-model"
        assert embedding.metadata["dimension"] == 3


class TestDatabaseManager:
    """Test database manager functionality."""
    
    @pytest.mark.asyncio
    async def test_database_initialization(self, database_manager):
        """Test database initialization."""
        assert database_manager is not None
        
        # Check if tables exist (would be implementation specific)
        # This is a placeholder for actual table existence checks
        
    @pytest.mark.asyncio
    async def test_document_operations(self, database_manager, sample_document_data):
        """Test document CRUD operations."""
        # Create document
        document_id = await database_manager.create_document(
            title=sample_document_data["title"],
            filename=sample_document_data["filename"],
            file_type=sample_document_data["file_type"],
            file_size=sample_document_data["file_size"],
            content=sample_document_data["content"],
            metadata=sample_document_data["metadata"]
        )
        
        assert document_id is not None
        
        # Read document
        document = await database_manager.get_document(document_id)
        assert document is not None
        assert document.title == sample_document_data["title"]
        assert document.filename == sample_document_data["filename"]
        
        # Update document
        await database_manager.update_document(
            document_id, 
            title="Updated Title"
        )
        
        updated_document = await database_manager.get_document(document_id)
        assert updated_document.title == "Updated Title"
        
        # List documents
        documents = await database_manager.list_documents()
        assert len(documents) >= 1
        
        # Delete document
        await database_manager.delete_document(document_id)
        deleted_document = await database_manager.get_document(document_id)
        assert deleted_document is None
        
    @pytest.mark.asyncio
    async def test_document_chunk_operations(self, database_manager, sample_document_data, sample_chunks):
        """Test document chunk operations."""
        # Create document first
        document_id = await database_manager.create_document(
            title=sample_document_data["title"],
            filename=sample_document_data["filename"],
            file_type=sample_document_data["file_type"],
            file_size=sample_document_data["file_size"],
            content=sample_document_data["content"],
            metadata=sample_document_data["metadata"]
        )
        
        # Create chunks
        chunk_ids = []
        for chunk_data in sample_chunks:
            chunk_id = await database_manager.create_document_chunk(
                document_id=document_id,
                content=chunk_data["content"],
                chunk_index=chunk_data["chunk_index"],
                start_char=chunk_data["start_char"],
                end_char=chunk_data["end_char"],
                metadata=chunk_data["metadata"]
            )
            chunk_ids.append(chunk_id)
            
        assert len(chunk_ids) == len(sample_chunks)
        
        # Get chunks for document
        chunks = await database_manager.get_document_chunks(document_id)
        assert len(chunks) == len(sample_chunks)
        
        # Verify chunk content
        for i, chunk in enumerate(chunks):
            assert chunk.content == sample_chunks[i]["content"]
            assert chunk.chunk_index == sample_chunks[i]["chunk_index"]
            
        # Clean up
        await database_manager.delete_document(document_id)
        
    @pytest.mark.asyncio
    async def test_conversation_operations(self, database_manager, sample_conversation_data):
        """Test conversation operations."""
        # Create conversation
        conversation_id = await database_manager.create_conversation(
            title=sample_conversation_data["title"]
        )
        
        assert conversation_id is not None
        
        # Add messages
        message_ids = []
        for message_data in sample_conversation_data["messages"]:
            message_id = await database_manager.add_message(
                conversation_id=conversation_id,
                role=message_data["role"],
                content=message_data["content"]
            )
            message_ids.append(message_id)
            
        # Get conversation with messages
        conversation = await database_manager.get_conversation_with_messages(conversation_id)
        assert conversation is not None
        assert len(conversation.messages) == len(sample_conversation_data["messages"])
        
        # Verify message content
        for i, message in enumerate(conversation.messages):
            assert message.role == sample_conversation_data["messages"][i]["role"]
            assert message.content == sample_conversation_data["messages"][i]["content"]
            
        # Clean up
        await database_manager.delete_conversation(conversation_id)
        
    @pytest.mark.asyncio
    async def test_embedding_operations(self, database_manager, sample_document_data, sample_chunks):
        """Test embedding storage and retrieval."""
        # Create document and chunk
        document_id = await database_manager.create_document(
            title=sample_document_data["title"],
            filename=sample_document_data["filename"],
            file_type=sample_document_data["file_type"],
            file_size=sample_document_data["file_size"],
            content=sample_document_data["content"],
            metadata=sample_document_data["metadata"]
        )
        
        chunk_id = await database_manager.create_document_chunk(
            document_id=document_id,
            content=sample_chunks[0]["content"],
            chunk_index=sample_chunks[0]["chunk_index"],
            start_char=sample_chunks[0]["start_char"],
            end_char=sample_chunks[0]["end_char"],
            metadata=sample_chunks[0]["metadata"]
        )
        
        # Store embedding
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        embedding_id = await database_manager.store_embedding(
            content_id=chunk_id,
            content_type="document_chunk",
            embedding_vector=test_embedding,
            model_name="test-model"
        )
        
        assert embedding_id is not None
        
        # Retrieve embedding
        embedding = await database_manager.get_embedding(embedding_id)
        assert embedding is not None
        assert embedding.content_id == chunk_id
        assert embedding.content_type == "document_chunk"
        assert embedding.embedding_vector == test_embedding
        assert embedding.model_name == "test-model"
        
        # Get embeddings by content
        embeddings = await database_manager.get_embeddings_by_content(chunk_id, "document_chunk")
        assert len(embeddings) >= 1
        
        # Clean up
        await database_manager.delete_document(document_id)
        
    @pytest.mark.asyncio
    async def test_search_functionality(self, database_manager, sample_document_data):
        """Test document search functionality."""
        # Create multiple documents
        doc_ids = []
        for i in range(3):
            doc_id = await database_manager.create_document(
                title=f"Test Document {i+1}",
                filename=f"test_{i+1}.txt",
                file_type="text/plain",
                file_size=1024,
                content=f"This is test document {i+1} with unique content about topic {i+1}.",
                metadata={"index": i+1}
            )
            doc_ids.append(doc_id)
            
        # Search documents by title
        results = await database_manager.search_documents("Test Document")
        assert len(results) >= 3
        
        # Search documents by content
        content_results = await database_manager.search_documents("unique content")
        assert len(content_results) >= 3
        
        # Search with filters
        filtered_results = await database_manager.search_documents(
            query="test", 
            file_types=["text/plain"]
        )
        assert len(filtered_results) >= 3
        
        # Clean up
        for doc_id in doc_ids:
            await database_manager.delete_document(doc_id)


class TestDatabasePerformance:
    """Test database performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_bulk_operations_performance(self, database_manager, performance_timer):
        """Test performance of bulk database operations."""
        performance_timer.start()
        
        # Create multiple documents
        doc_ids = []
        for i in range(100):
            doc_id = await database_manager.create_document(
                title=f"Bulk Document {i}",
                filename=f"bulk_{i}.txt",
                file_type="text/plain",
                file_size=1024,
                content=f"Bulk document content {i}",
                metadata={"bulk_index": i}
            )
            doc_ids.append(doc_id)
            
        performance_timer.stop()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert performance_timer.elapsed < 10.0  # 10 seconds max
        assert len(doc_ids) == 100
        
        # Clean up
        for doc_id in doc_ids:
            await database_manager.delete_document(doc_id)
            
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, database_manager):
        """Test concurrent database operations."""
        # Create tasks for concurrent document creation
        async def create_document(index):
            return await database_manager.create_document(
                title=f"Concurrent Document {index}",
                filename=f"concurrent_{index}.txt",
                file_type="text/plain",
                file_size=1024,
                content=f"Concurrent document content {index}",
                metadata={"concurrent_index": index}
            )
            
        # Run concurrent operations
        tasks = [create_document(i) for i in range(10)]
        doc_ids = await asyncio.gather(*tasks)
        
        assert len(doc_ids) == 10
        assert all(doc_id is not None for doc_id in doc_ids)
        
        # Clean up
        for doc_id in doc_ids:
            await database_manager.delete_document(doc_id)


class TestDatabaseErrorHandling:
    """Test database error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_document_operations(self, database_manager):
        """Test operations with invalid document IDs."""
        # Try to get non-existent document
        document = await database_manager.get_document("invalid_id")
        assert document is None
        
        # Try to update non-existent document
        result = await database_manager.update_document("invalid_id", title="New Title")
        assert result is False
        
        # Try to delete non-existent document
        result = await database_manager.delete_document("invalid_id")
        assert result is False
        
    @pytest.mark.asyncio
    async def test_constraint_violations(self, database_manager, sample_document_data):
        """Test database constraint violations."""
        # Create document
        doc_id = await database_manager.create_document(
            title=sample_document_data["title"],
            filename=sample_document_data["filename"],
            file_type=sample_document_data["file_type"],
            file_size=sample_document_data["file_size"],
            content=sample_document_data["content"],
            metadata=sample_document_data["metadata"]
        )
        
        # Try to create chunk with invalid document_id
        with pytest.raises(Exception):  # Should raise foreign key constraint error
            await database_manager.create_document_chunk(
                document_id="invalid_document_id",
                content="Test chunk",
                chunk_index=0,
                start_char=0,
                end_char=10
            )
            
        # Clean up
        await database_manager.delete_document(doc_id)
        
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, database_manager):
        """Test transaction rollback on errors."""
        # This would test database transaction handling
        # Implementation depends on the specific database manager
        pass