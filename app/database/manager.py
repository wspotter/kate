"""
Database manager for Kate LLM Client with async SQLAlchemy 2.0.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator, Dict, Any
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from sqlalchemy import text
from alembic.config import Config
from alembic import command
from loguru import logger

from .models import Base
from ..core.config import DatabaseSettings


class DatabaseManager:
    """
    Manages database connections and operations using SQLAlchemy 2.0.
    
    Provides async database operations, connection pooling, and migration support.
    """
    
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings
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
            engine_kwargs = {
                "echo": self.settings.echo,
                "future": True,
            }
            
            # Special handling for SQLite
            if self.settings.url.startswith("sqlite"):
                engine_kwargs.update({
                    "poolclass": StaticPool,
                    "pool_pre_ping": True,
                    "connect_args": {
                        "check_same_thread": False,
                        "timeout": 20,
                    }
                })
            else:
                engine_kwargs.update({
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