"""Database manager for Kate LLM Client with async SQLAlchemy 2.0.

Adds legacy CRUD helper methods expected by older tests.
"""
import hashlib
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

# (Alembic imports removed – not used in current lightweight test context)
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from ..core.config import DatabaseSettings
from .models import Base, ContentType, Conversation, Document, DocumentChunk, Message


class DatabaseManager:
    """
    Manages database connections and operations using SQLAlchemy 2.0.
    
    Provides async database operations, connection pooling, and migration support.
    """
    
    def __init__(self, settings: DatabaseSettings | str):
        if isinstance(settings, str):  # Legacy: raw URL
            settings = DatabaseSettings(url=settings)  # type: ignore[arg-type]
        self.settings: DatabaseSettings = settings
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self.logger = logger.bind(component="DatabaseManager")
        
    async def initialize(self) -> None:
        """
        Initialize the database engine and session factory.
        """
        if self.engine is not None:
            self.logger.warning("Database already initialized")
            return
            
        try:
            self.logger.info("Initializing database connection...")
            
            # Create engine with appropriate settings
            engine_kwargs: Dict[str, Any] = {
                "echo": self.settings.echo,
                "future": True,
            }
            
            # Special handling for SQLite
            if self.settings.url.startswith("sqlite"):
                engine_kwargs.update({  # type: ignore[arg-type]
                    "poolclass": StaticPool,
                    "pool_pre_ping": True,
                    "connect_args": {"check_same_thread": False, "timeout": 20},
                })
            else:
                engine_kwargs.update({  # type: ignore[arg-type]
                    "pool_size": self.settings.pool_size,
                    "max_overflow": self.settings.max_overflow,
                    "pool_pre_ping": True,
                })
            
            self.engine = create_async_engine(self.settings.url, **engine_kwargs)
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test the connection
            await self._test_connection()
            
            # Auto-create tables (legacy tests expect tables without explicit migration call)
            await self.create_tables()
            self.logger.info("Database connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
            
    async def _test_connection(self) -> None:
        """Test the database connection."""
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            self.logger.debug("Database connection test passed")
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            raise
            
    async def create_tables(self) -> None:
        """
        Create all database tables.
        """
        if not self.engine:
            raise RuntimeError("Database not initialized")
            
        try:
            self.logger.info("Creating database tables...")
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise
            
    async def drop_tables(self) -> None:
        """
        Drop all database tables. Use with caution!
        """
        if not self.engine:
            raise RuntimeError("Database not initialized")
            
        try:
            self.logger.warning("Dropping all database tables...")
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            self.logger.warning("All database tables dropped")
        except Exception as e:
            self.logger.error(f"Failed to drop tables: {e}")
            raise
            
    async def migrate(self) -> None:
        """
        Run database migrations using Alembic.
        """
        try:
            self.logger.info("Running database migrations...")
            
            # For now, just create tables if they don't exist
            # TODO: Implement proper Alembic migrations
            await self.create_tables()
            
            self.logger.info("Database migrations completed")
        except Exception as e:
            self.logger.error(f"Database migration failed: {e}")
            raise
            
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Async context manager for database sessions.
        
        Usage:
            async with db_manager.session() as session:
                result = await session.execute(query)
                await session.commit()
        """
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
            
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
                
    async def execute_raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute raw SQL query.
        
        Args:
            sql: SQL query string
            params: Optional query parameters
            
        Returns:
            Query result
        """
        async with self.session() as session:
            result = await session.execute(text(sql), params or {})
            return result
            
    async def get_table_info(self) -> Dict[str, Any]:
        """
        Get information about database tables.
        
        Returns:
            Dictionary with table information
        """
        if not self.engine:
            raise RuntimeError("Database not initialized")
            
        tables_info = {}
        
        try:
            async with self.session() as session:
                # Get table names
                if self.settings.url.startswith("sqlite"):
                    result = await session.execute(
                        text("SELECT name FROM sqlite_master WHERE type='table'")
                    )
                    tables = [row[0] for row in result.fetchall()]
                else:
                    # PostgreSQL, MySQL, etc.
                    result = await session.execute(
                        text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
                    )
                    tables = [row[0] for row in result.fetchall()]
                
                # Get row counts for each table
                for table in tables:
                    try:
                        count_result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = count_result.scalar()
                        tables_info[table] = {"row_count": count}
                    except Exception as e:
                        tables_info[table] = {"error": str(e)}
                        
        except Exception as e:
            self.logger.error(f"Failed to get table info: {e}")
            raise
            
        return tables_info
        
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the database connection.
        """
        if self.engine:
            try:
                self.logger.info("Shutting down database connection...")
                await self.engine.dispose()
                self.engine = None
                self.session_factory = None
                self.logger.info("Database connection shut down successfully")
            except Exception as e:
                self.logger.error(f"Error during database shutdown: {e}")
        else:
            self.logger.debug("Database not initialized, nothing to shutdown")
            
    def is_initialized(self) -> bool:
        """Check if the database is initialized."""
        return self.engine is not None and self.session_factory is not None

    # ------------------------------------------------------------------
    # Legacy CRUD methods expected by tests
    # ------------------------------------------------------------------
    async def create_document(
        self,
        title: str,
        filename: str,
        file_type: str,
        file_size: int,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
        # Basic hash (sha256 of content)
        doc_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        document = Document(
            title=title,
            content=content,
            content_type=ContentType.DOCUMENT,
            doc_hash=doc_hash,
            word_count=len(content.split()),
            extra_data=metadata or {},
            source_path=filename,
        )
        async with self.session() as session:
            session.add(document)
            await session.commit()
            return document.id

    async def get_document(self, document_id: str) -> Optional[Document]:
        async with self.session() as session:
            return await session.get(Document, document_id)

    async def update_document(self, document_id: str, **updates: Any) -> bool:
        async with self.session() as session:
            doc = await session.get(Document, document_id)
            if not doc:
                return False
            for k, v in updates.items():
                if k == "metadata":
                    doc.extra_data = v
                elif hasattr(doc, k):
                    setattr(doc, k, v)
            await session.commit()
            return True

    async def list_documents(self) -> List[Document]:
        async with self.session() as session:
            result = await session.execute(text("SELECT id FROM documents"))
            ids = [row[0] for row in result.fetchall()]
            docs = []
            for did in ids:
                d = await session.get(Document, did)
                if d:
                    docs.append(d)
            return docs

    async def delete_document(self, document_id: str) -> bool:
        async with self.session() as session:
            doc = await session.get(Document, document_id)
            if not doc:
                return False
            await session.delete(doc)
            await session.commit()
            return True

    async def create_document_chunk(
        self,
        document_id: str,
        content: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        chunk = DocumentChunk(
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            extra_data=metadata or {},
            word_count=len(content.split()),
        )
        async with self.session() as session:
            session.add(chunk)
            await session.commit()
            return chunk.id

    async def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        async with self.session() as session:
            result = await session.execute(text("SELECT id FROM document_chunks WHERE document_id = :did"), {"did": document_id})
            ids = [row[0] for row in result.fetchall()]
            chunks: list[DocumentChunk] = []
            for cid in ids:
                c = await session.get(DocumentChunk, cid)
                if c:
                    chunks.append(c)
            return chunks

    async def create_conversation(self, title: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        conv = Conversation(title=title, extra_data=metadata or {})
        async with self.session() as session:
            session.add(conv)
            await session.commit()
            return conv.id

    async def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        msg = Message(conversation_id=conversation_id, role=role, content=content, extra_data=metadata or {})
        async with self.session() as session:
            session.add(msg)
            await session.commit()
            return msg.id

    async def get_conversation_with_messages(self, conversation_id: str) -> Optional[Conversation]:
        async with self.session() as session:
            conv = await session.get(Conversation, conversation_id)
            if conv:
                # Load messages (lazy load triggers SELECT)
                _ = conv.messages  # access to force load
            return conv

    async def delete_conversation(self, conversation_id: str) -> bool:
        async with self.session() as session:
            conv = await session.get(Conversation, conversation_id)
            if not conv:
                return False
            await session.delete(conv)
            await session.commit()
            return True