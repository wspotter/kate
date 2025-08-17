"""
Background processing service for document indexing and other long-running operations.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from PySide6.QtCore import QObject, QThread, QTimer, Signal

from ..database.models import Document
from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .vector_store import VectorStore


class TaskStatus(Enum):
    """Status of background tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Types of background tasks."""
    DOCUMENT_INDEXING = "document_indexing"
    DOCUMENT_REINDEXING = "document_reindexing"
    BULK_INDEXING = "bulk_indexing"
    VECTOR_OPTIMIZATION = "vector_optimization"


@dataclass
class BackgroundTask:
    """Represents a background task."""
    task_id: str
    task_type: TaskType
    status: TaskStatus
    document_id: Optional[str] = None
    document_name: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    error_message: str = ""
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class DocumentIndexingWorker(QThread):
    """Worker thread for document indexing operations."""
    
    # Signals
    progress_updated = Signal(str, int, str)  # task_id, completed_steps, status_text
    task_completed = Signal(str, bool, str)  # task_id, success, message
    error_occurred = Signal(str, str)  # task_id, error_message
    
    def __init__(self, 
                 task: BackgroundTask,
                 document_processor: DocumentProcessor,
                 embedding_service: EmbeddingService,
                 vector_store: VectorStore):
        super().__init__()
        self.task = task
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.logger = logger.bind(component="DocumentIndexingWorker")
        self._cancelled = False
        
    def run(self) -> None:
        """Run the document indexing task."""
        try:
            self.logger.info(f"Starting document indexing task: {self.task.task_id}")
            
            if self.task.task_type == TaskType.DOCUMENT_INDEXING:
                self._process_document_indexing()
            elif self.task.task_type == TaskType.DOCUMENT_REINDEXING:
                self._process_document_reindexing()
            elif self.task.task_type == TaskType.BULK_INDEXING:
                self._process_bulk_indexing()
            else:
                raise ValueError(f"Unsupported task type: {self.task.task_type}")
                
        except Exception as e:
            self.logger.error(f"Error in document indexing task {self.task.task_id}: {e}")
            self.error_occurred.emit(self.task.task_id, str(e))
            
    def _process_document_indexing(self) -> None:
        """Process single document indexing."""
        if self._cancelled:
            return
            
        # Step 1: Load document
        self.progress_updated.emit(self.task.task_id, 1, "Loading document...")
        
        # Simulate document loading (replace with actual implementation)
        self.msleep(500)  # Simulate processing time
        
        if self._cancelled:
            return
            
        # Step 2: Extract text and create chunks
        self.progress_updated.emit(self.task.task_id, 2, "Extracting text...")
        
        # Simulate text extraction
        chunks = self._simulate_text_extraction()
        self.task.total_steps = len(chunks) + 3  # +3 for loading, extraction, and saving
        
        if self._cancelled:
            return
            
        # Step 3: Process chunks and generate embeddings
        for i, chunk in enumerate(chunks):
            if self._cancelled:
                return
                
            self.progress_updated.emit(
                self.task.task_id, 
                3 + i, 
                f"Processing chunk {i+1}/{len(chunks)}..."
            )
            
            # Simulate embedding generation
            self.msleep(100)  # Simulate processing time per chunk
            
        # Step 4: Save to vector store
        self.progress_updated.emit(self.task.task_id, self.task.total_steps, "Saving to vector store...")
        self.msleep(300)  # Simulate saving time
        
        if not self._cancelled:
            self.task_completed.emit(self.task.task_id, True, "Document indexed successfully")
            
    def _process_document_reindexing(self) -> None:
        """Process document reindexing."""
        if self._cancelled:
            return
            
        # Step 1: Remove existing embeddings
        self.progress_updated.emit(self.task.task_id, 1, "Removing existing embeddings...")
        self.msleep(200)
        
        # Step 2: Reprocess document
        self._process_document_indexing()
        
    def _process_bulk_indexing(self) -> None:
        """Process bulk document indexing."""
        # This would process multiple documents
        # For now, simulate with single document processing
        self._process_document_indexing()
        
    def _simulate_text_extraction(self) -> List[str]:
        """Simulate text extraction returning chunks."""
        # In real implementation, this would use document_processor
        return [f"Text chunk {i}" for i in range(10, 50)]  # Variable number of chunks
        
    def cancel(self) -> None:
        """Cancel the current task."""
        self._cancelled = True
        self.logger.info(f"Cancelling task: {self.task.task_id}")


class BackgroundProcessingService(QObject):
    """Service for managing background processing tasks."""
    
    # Signals
    task_added = Signal(str, str, str, int)  # task_id, document_id, document_name, total_steps
    task_progress_updated = Signal(str, int, str)  # task_id, completed_steps, status_text
    task_completed = Signal(str, bool, str)  # task_id, success, message
    task_error = Signal(str, str)  # task_id, error_message
    queue_status_changed = Signal(int, int)  # active_tasks, queued_tasks
    
    def __init__(self,
                 document_processor: DocumentProcessor,
                 embedding_service: EmbeddingService,
                 vector_store: VectorStore):
        super().__init__()
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.logger = logger.bind(component="BackgroundProcessingService")
        
        # Task management
        self.tasks: Dict[str, BackgroundTask] = {}
        self.active_workers: Dict[str, DocumentIndexingWorker] = {}
        self.task_queue: List[str] = []  # Queue of task IDs
        self.max_concurrent_tasks = 2  # Maximum concurrent indexing tasks
        
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._emit_queue_status)
        self.status_timer.start(1000)  # Update every second
        
    def start_document_indexing(self, 
                               document_id: str, 
                               document_name: str,
                               task_type: TaskType = TaskType.DOCUMENT_INDEXING) -> str:
        """Start document indexing task."""
        task_id = str(uuid.uuid4())
        
        task = BackgroundTask(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            document_id=document_id,
            document_name=document_name,
            total_steps=0  # Will be updated during processing
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        
        self.logger.info(f"Queued document indexing task: {document_name} ({task_id})")
        
        # Try to start the task immediately
        self._process_queue()
        
        return task_id
        
    def start_bulk_indexing(self, document_list: List[tuple]) -> List[str]:
        """Start bulk document indexing.
        
        Args:
            document_list: List of (document_id, document_name) tuples
            
        Returns:
            List of task IDs
        """
        task_ids = []
        
        for document_id, document_name in document_list:
            task_id = self.start_document_indexing(
                document_id, 
                document_name, 
                TaskType.BULK_INDEXING
            )
            task_ids.append(task_id)
            
        self.logger.info(f"Started bulk indexing for {len(document_list)} documents")
        return task_ids
        
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task."""
        if task_id not in self.tasks:
            self.logger.warning(f"Task not found: {task_id}")
            return False
            
        task = self.tasks[task_id]
        
        if task.status == TaskStatus.RUNNING:
            # Cancel running task
            if task_id in self.active_workers:
                self.active_workers[task_id].cancel()
                self.active_workers[task_id].wait()  # Wait for thread to finish
                del self.active_workers[task_id]
                
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow()
            
        elif task.status == TaskStatus.PENDING:
            # Remove from queue
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
                
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow()
            
        self.logger.info(f"Cancelled task: {task_id}")
        self._emit_queue_status()
        return True
        
    def get_task_status(self, task_id: str) -> Optional[BackgroundTask]:
        """Get status of a specific task."""
        return self.tasks.get(task_id)
        
    def get_active_tasks(self) -> List[BackgroundTask]:
        """Get list of currently active tasks."""
        return [task for task in self.tasks.values() 
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]]
                
    def get_completed_tasks(self) -> List[BackgroundTask]:
        """Get list of completed tasks."""
        return [task for task in self.tasks.values()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]]
                
    def clear_completed_tasks(self) -> None:
        """Clear all completed tasks from memory."""
        completed_task_ids = [
            task_id for task_id, task in self.tasks.items()
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        ]
        
        for task_id in completed_task_ids:
            del self.tasks[task_id]
            
        self.logger.info(f"Cleared {len(completed_task_ids)} completed tasks")
        self._emit_queue_status()
        
    def _process_queue(self) -> None:
        """Process the task queue."""
        if len(self.active_workers) >= self.max_concurrent_tasks:
            return  # Already at maximum capacity
            
        if not self.task_queue:
            return  # No tasks in queue
            
        # Start next task in queue
        task_id = self.task_queue.pop(0)
        task = self.tasks[task_id]
        
        if task.status != TaskStatus.PENDING:
            # Task was cancelled or modified, skip it
            self._process_queue()  # Try next task
            return
            
        # Create and start worker
        worker = DocumentIndexingWorker(
            task, 
            self.document_processor, 
            self.embedding_service, 
            self.vector_store
        )
        
        # Connect worker signals
        worker.progress_updated.connect(self._on_worker_progress)
        worker.task_completed.connect(self._on_worker_completed)
        worker.error_occurred.connect(self._on_worker_error)
        worker.finished.connect(lambda: self._on_worker_finished(task_id))
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        
        # Store worker and start
        self.active_workers[task_id] = worker
        worker.start()
        
        # Emit signals
        self.task_added.emit(task_id, task.document_id or "", task.document_name, task.total_steps)
        
        self.logger.info(f"Started processing task: {task_id}")
        
        # Try to start more tasks if queue has items
        self._process_queue()
        
    def _on_worker_progress(self, task_id: str, completed_steps: int, status_text: str) -> None:
        """Handle worker progress updates."""
        if task_id in self.tasks:
            self.tasks[task_id].completed_steps = completed_steps
            self.task_progress_updated.emit(task_id, completed_steps, status_text)
            
    def _on_worker_completed(self, task_id: str, success: bool, message: str) -> None:
        """Handle worker completion."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            if not success:
                task.error_message = message
                
            self.task_completed.emit(task_id, success, message)
            
    def _on_worker_error(self, task_id: str, error_message: str) -> None:
        """Handle worker errors."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.error_message = error_message
            
            self.task_error.emit(task_id, error_message)
            
    def _on_worker_finished(self, task_id: str) -> None:
        """Handle worker thread cleanup."""
        if task_id in self.active_workers:
            del self.active_workers[task_id]
            
        # Process next task in queue
        self._process_queue()
        
        self._emit_queue_status()
        
    def _emit_queue_status(self) -> None:
        """Emit current queue status."""
        active_count = len(self.active_workers)
        queued_count = len(self.task_queue)
        self.queue_status_changed.emit(active_count, queued_count)