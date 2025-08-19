"""
Chat area component for Kate LLM Client with RAG integration.
"""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    # QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    # QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...core.events import EventBus
from ...providers.base import ChatCompletionRequest, ChatMessage
from ...services.rag_evaluation_service import RAGEvaluationService
from .message_bubble import MessageBubble, StreamingMessageBubble


class RAGWorker(QObject):
    """Worker for handling RAG processing in separate thread."""
    
    # Signals
    response_ready = Signal(str)  # RAG response content
    context_ready = Signal(list)  # Retrieved context sources
    error_occurred = Signal(str)  # Error message
    streaming_chunk = Signal(str)  # Streaming response chunk

    def __init__(self, rag_integration_service):
        super().__init__()
        self.rag_integration_service = rag_integration_service
        
    def process_message(self, conversation_id: str, user_message: str):
        """Process message with RAG integration."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Process the message
            response = loop.run_until_complete(
                self.rag_integration_service.process_message(conversation_id, user_message)
            )
            
            # Emit results
            self.response_ready.emit(response.content)
            self.context_ready.emit(response.retrieved_sources)
            
            loop.close()
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            
    def process_message_streaming(self, conversation_id: str, user_message: str):
        """Process message with streaming RAG response."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def stream_response():
                async for chunk in self.rag_integration_service.process_message_streaming(
                    conversation_id, user_message, lambda x: self.streaming_chunk.emit(x)
                ):
                    pass  # Chunks are emitted via callback
                    
            loop.run_until_complete(stream_response())
            loop.close()
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class ChatScrollArea(QScrollArea):
    """Custom scroll area for chat messages."""
    
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(16, 16, 16, 16)
        self.content_layout.setSpacing(12)
        self.content_layout.addStretch()  # Push messages to bottom initially
        
        self.setWidget(self.content_widget)
        
    def add_message_widget(self, message_widget: MessageBubble) -> None:
        """Add a message widget to the chat area."""
        # Insert before the stretch
        count = self.content_layout.count()
        self.content_layout.insertWidget(count - 1, message_widget)
        
        # Auto-scroll to bottom
        QTimer.singleShot(10, self._scroll_to_bottom)
        
    def _scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the chat area."""
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def clear_messages(self) -> None:
        """Clear all messages from the chat area."""
        # Remove all widgets except the stretch
        while self.content_layout.count() > 1:
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


class MessageInputArea(QWidget):
    """Message input area with send button."""
    
    # Signals
    message_sent = Signal(str)
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self) -> None:
        """Set up the input area UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 16)
        layout.setSpacing(8)
        
        # Message input
        self.message_input = QTextEdit()
        self.message_input.setPlaceholderText("Type your message here... (Ctrl+Enter to send)")
        self.message_input.setMaximumHeight(120)
        self.message_input.setMinimumHeight(40)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.setFixedSize(80, 40)
        self.send_button.setEnabled(False)
        
        layout.addWidget(self.message_input, 1)
        layout.addWidget(self.send_button)
        
        # Apply styling
        self._apply_styling()
        
    def _apply_styling(self) -> None:
        """Apply styling to the input area."""
        self.setStyleSheet("""
            MessageInputArea {
                background-color: #2b2b2b;
                border-top: 1px solid #404040;
            }
            
            QTextEdit {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 12px;
                color: #ffffff;
                font-size: 14px;
                line-height: 1.4;
            }
            
            QTextEdit:focus {
                border-color: #0078d4;
            }
            
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 8px;
                color: #ffffff;
                font-weight: bold;
                font-size: 14px;
            }
            
            QPushButton:hover {
                background-color: #106ebe;
            }
            
            QPushButton:pressed {
                background-color: #005a9e;
            }
            
            QPushButton:disabled {
                background-color: #404040;
                color: #888888;
            }
        """)
        
    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.send_button.clicked.connect(self._send_message)
        self.message_input.textChanged.connect(self._on_text_changed)
        
    def _send_message(self) -> None:
        """Send the current message."""
        text = self.message_input.toPlainText().strip()
        if text:
            self.message_sent.emit(text)
            self.message_input.clear()
            
    def _on_text_changed(self) -> None:
        """Handle text input changes."""
        has_text = bool(self.message_input.toPlainText().strip())
        self.send_button.setEnabled(has_text)
        
    def keyPressEvent(self, event) -> None:
        """Handle key press events."""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if event.modifiers() & Qt.ControlModifier:
                # Ctrl+Enter sends message
                self._send_message()
                return
        super().keyPressEvent(event)
        
    def set_focus(self) -> None:
        """Set focus to the message input."""
        self.message_input.setFocus()


class ChatArea(QWidget):
    """
    Main chat area widget for displaying conversations and message input with RAG integration.
    """
    
    # Signals
    message_sent = Signal(str)
    rag_context_updated = Signal(list)  # Emitted when RAG context is updated
    evaluation_received = Signal(object)  # Emitted when evaluation is received
    
    def __init__(self, event_bus: EventBus, rag_integration_service=None, evaluation_service: Optional[RAGEvaluationService] = None):
        super().__init__()
        self.event_bus = event_bus
        self.rag_integration_service = rag_integration_service
        self.evaluation_service = evaluation_service
        self.logger = logger.bind(component="ChatArea")
        
        self.current_conversation_id: Optional[str] = None
        self.messages: List[MessageBubble] = []
        self.current_streaming_message: Optional[StreamingMessageBubble] = None
        # Assistant & model settings
        self.active_assistant_id: Optional[str] = None
        self.assistant_system_prompt: Optional[str] = None
        self.model_settings: Dict[str, Any] = {"temperature": 0.7, "stream": False}

        # RAG worker and thread
        self.rag_worker = None
        self.rag_thread = None

        self.setAcceptDrops(True)
        self._setup_ui()
        self._connect_signals()

        # Initialize RAG worker if service is available
        if self.rag_integration_service:
            self._setup_rag_worker()

        # Add drop overlay
        self.drop_overlay = DropOverlay(self)
        
    def _setup_ui(self) -> None:
        """Set up the chat area UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        self.header = self._create_header()
        layout.addWidget(self.header)
        
        # Chat scroll area
        self.chat_scroll = ChatScrollArea()
        layout.addWidget(self.chat_scroll, 1)
        
        # Message input area
        self.input_area = MessageInputArea()
        layout.addWidget(self.input_area)
        
        # Apply styling
        self._apply_styling()
        
    def _create_header(self) -> QWidget:
        """Create the chat header."""
        header = QFrame()
        header.setFixedHeight(60)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)
        
        # Title and info
        self.title_label = QLabel("Select a conversation")
        self.title_label.setFont(QFont("Arial", 14, QFont.Bold))
        
        self.info_label = QLabel("")
        self.info_label.setFont(QFont("Arial", 10))
        
        # RAG status indicator
        self.rag_status_label = QLabel("RAG: Ready")
        self.rag_status_label.setFont(QFont("Arial", 9))
        self.rag_status_label.setStyleSheet("color: #00ff00;")  # Green when ready
        
        title_layout = QVBoxLayout()
        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.info_label)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        layout.addWidget(self.rag_status_label)
        
        return header
        
    def _apply_styling(self) -> None:
        """Apply styling to the chat area."""
        self.setStyleSheet("""
            ChatArea {
                background-color: #1e1e1e;
            }
            
            QFrame {
                background-color: #2b2b2b;
                border-bottom: 1px solid #404040;
            }
            
            QLabel {
                color: #ffffff;
            }
        """)
        
    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.input_area.message_sent.connect(self._handle_message_sent)
        
    def _setup_rag_worker(self) -> None:
        """Set up RAG worker thread."""
        self.rag_thread = QThread()
        self.rag_worker = RAGWorker(self.rag_integration_service)
        self.rag_worker.moveToThread(self.rag_thread)
        
        # Connect signals
        self.rag_worker.response_ready.connect(self._handle_rag_response)
        self.rag_worker.context_ready.connect(self._handle_rag_context)
        self.rag_worker.error_occurred.connect(self._handle_rag_error)
        self.rag_worker.streaming_chunk.connect(self._handle_streaming_chunk)
        
        self.rag_thread.start()
        
        self.logger.info("RAG worker thread initialized")
        
    def _handle_message_sent(self, message: str) -> None:
        """Handle message sent from input area."""
        # Auto-create a default conversation if none selected
        if not self.current_conversation_id:
            self.load_conversation("default")
        
        # Add user message to chat
        self.add_message("user", message)
        
        # Emit signal for external handling
        self.message_sent.emit(message)
        
        # Prefer RAG if configured
        if self.rag_integration_service and self.rag_worker:
            self._process_with_rag(message)
            return
        
        # Otherwise try Ollama provider through application reference (walk parent chain)
        app_ref = self._get_application()
        if app_ref and getattr(app_ref, "ollama_provider", None) and app_ref.ollama_provider.is_connected and app_ref.selected_model:
            asyncio.create_task(self._ollama_chat(message, app_ref))
        else:
            self._add_placeholder_response()

    def _get_application(self):
        """Attempt to access the KateApplication instance via parent widgets."""
        p = self.parent()
        depth = 0
        while p and depth < 10:
            if hasattr(p, "app") and getattr(p, "app", None):  # MainWindow holds app
                return getattr(p, "app")
            p = p.parent()
            depth += 1
        return None

    async def _ollama_chat(self, user_text: str, app_ref) -> None:
        """Send message to Ollama provider (non-streaming fallback for reliability)."""
        provider = app_ref.ollama_provider
        model = app_ref.selected_model
        if not provider or not model:
            self._add_placeholder_response()
            return
        try:
            self.set_loading(True)
            # Build conversation messages (simple: user only for now; could retain history later)
            system_prompt = self.assistant_system_prompt or "You are a helpful local AI."
            chat_messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_text),
            ]
            # Direct non-streaming request
            request = ChatCompletionRequest(
                model=model,
                messages=chat_messages,
                stream=self.model_settings.get("stream", False),
                temperature=self.model_settings.get("temperature", 0.7),
                max_tokens=self.model_settings.get("max_tokens"),
                top_p=self.model_settings.get("top_p", 1.0),
                frequency_penalty=self.model_settings.get("frequency_penalty", 0.0),
                presence_penalty=self.model_settings.get("presence_penalty", 0.0)
            )
            import time
            start_t = time.time()
            resp = await provider.chat_completion(request)
            elapsed = time.time() - start_t
            content = resp.choices[0]["message"]["content"] if resp.choices else "<no content>"
            # Create stub evaluation metrics if evaluation service exists
            eval_obj = None
            try:
                from datetime import datetime

                from ...services.rag_evaluation_service import (
                    ResponseEvaluation,
                    RetrievalContext,
                )
                eval_obj = ResponseEvaluation(
                    evaluation_id=str(int(start_t * 1000)),
                    timestamp=datetime.now(),
                    query=user_text,
                    response=content,
                    retrieval_context=RetrievalContext(
                        document_chunks=[],
                        similarity_scores=[],
                        retrieval_query=user_text,
                        total_retrieved=0,
                        retrieval_time=0.0,
                    ),
                    relevance_score=0.5,
                    coherence_score=0.5,
                    completeness_score=0.5,
                    citation_accuracy=0.0,
                    answer_quality=0.5,
                    factual_accuracy=0.5,
                    response_time=elapsed,
                    overall_score=0.5,
                    confidence_score=0.5,
                )
            except Exception as ee:
                self.logger.debug(f"Could not build evaluation stub: {ee}")

            # Pass a simplified dict for bubble evaluation display
            eval_dict = None
            if eval_obj:
                eval_dict = {
                    "overall_score": eval_obj.overall_score,
                    "relevance_score": eval_obj.relevance_score,
                    "coherence_score": eval_obj.coherence_score,
                    "completeness_score": eval_obj.completeness_score,
                }
            self.add_message("assistant", content, evaluation_data=eval_dict)
            if eval_obj:
                # Emit full evaluation object for panel/dashboard
                self.evaluation_received.emit(eval_obj)
        except Exception as e:
            self.logger.error(f"Ollama chat failed: {e}")
            self.add_message("assistant", f"Error contacting Ollama: {e}")
        finally:
            self.set_loading(False)

    def set_assistant(self, assistant_id: str, assistant_data: Dict[str, Any]) -> None:
        """Configure active assistant (updates system prompt)."""
        self.active_assistant_id = assistant_id
        # Use description as system prompt base
        system_prompt = assistant_data.get("system_prompt")
        if not system_prompt:
            desc = assistant_data.get("description") or assistant_data.get("name") or "Assistant"
            system_prompt = f"You are '{assistant_data.get('name', 'Assistant')}'. {desc}"
        self.assistant_system_prompt = system_prompt
        # Adjust temperature heuristics per assistant role
        role_temps = {
            'creative': 0.9,
            'coding': 0.55,
            'data': 0.45,
            'research': 0.6,
            'general': 0.7
        }
        if assistant_id in role_temps:
            self.model_settings['temperature'] = role_temps[assistant_id]
        self.logger.debug(f"Assistant context set: {assistant_id}")

    def set_model_settings(self, settings: Dict[str, Any]) -> None:
        """Update chat model parameter settings."""
        self.model_settings.update(settings)
        self.logger.debug(f"Model settings updated in chat area: {self.model_settings}")
        
    def _process_with_rag(self, message: str) -> None:
        """Process message with RAG integration."""
        try:
            # Update status
            self.set_loading(True)
            self.rag_status_label.setText("RAG: Processing...")
            self.rag_status_label.setStyleSheet("color: #ffaa00;")  # Orange when processing
            
            # Create streaming message bubble for assistant response
            self.current_streaming_message = StreamingMessageBubble("assistant")
            self.messages.append(self.current_streaming_message)
            self.chat_scroll.add_message_widget(self.current_streaming_message)
            
            # Process message in worker thread (use streaming)
            QTimer.singleShot(10, lambda: self.rag_worker.process_message_streaming(
                self.current_conversation_id, message
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to process with RAG: {e}")
            self._handle_rag_error(str(e))
            
    def _handle_rag_response(self, response_content: str) -> None:
        """Handle RAG response."""
        try:
            # Finish streaming message if active
            if self.current_streaming_message:
                self.current_streaming_message.finish_streaming()
                
                # Check if there's evaluation data in the response metadata
                # This would be available from the RAG integration service
                if hasattr(self.current_streaming_message, 'pending_evaluation') and self.current_streaming_message.pending_evaluation:
                    # Emit evaluation signal for assistant panel updates
                    self.evaluation_received.emit(self.current_streaming_message.pending_evaluation)
                
                self.current_streaming_message = None
            else:
                # Add as regular message if no streaming
                self.add_message("assistant", response_content)
            
            # Update status
            self.set_loading(False)
            self.rag_status_label.setText("RAG: Ready")
            self.rag_status_label.setStyleSheet("color: #00ff00;")  # Green when ready
            
            self.logger.info("RAG response processed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to handle RAG response: {e}")
            
    def _handle_rag_context(self, sources: list) -> None:
        """Handle RAG context sources."""
        try:
            self.rag_context_updated.emit(sources)
            self.logger.debug(f"RAG context updated with {len(sources)} sources")
        except Exception as e:
            self.logger.error(f"Failed to handle RAG context: {e}")
            
    def _handle_rag_error(self, error_message: str) -> None:
        """Handle RAG processing error."""
        try:
            # Finish streaming message if active
            if self.current_streaming_message:
                self.current_streaming_message.append_content(f"\n\nError: {error_message}")
                self.current_streaming_message.finish_streaming()
                self.current_streaming_message = None
            else:
                # Add error message
                self.add_message("assistant", f"Sorry, I encountered an error: {error_message}")
            
            # Update status
            self.set_loading(False)
            self.rag_status_label.setText("RAG: Error")
            self.rag_status_label.setStyleSheet("color: #ff0000;")  # Red when error
            
            self.logger.error(f"RAG processing error: {error_message}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle RAG error: {e}")
            
    def _handle_streaming_chunk(self, chunk: str) -> None:
        """Handle streaming response chunk."""
        try:
            if self.current_streaming_message:
                self.current_streaming_message.append_content(chunk)
        except Exception as e:
            self.logger.error(f"Failed to handle streaming chunk: {e}")
        
    def _add_placeholder_response(self) -> None:
        """Add a placeholder AI response when RAG is not available."""
        placeholder_responses = [
            "I understand your question. Let me help you with that.",
            "That's an interesting point. Here's my perspective...",
            "I'd be happy to assist you with this task.",
            "Let me think about this and provide you with a detailed response."
        ]
        
        import random
        response = random.choice(placeholder_responses)
        
        # Add after a short delay to simulate processing
        QTimer.singleShot(1000, lambda: self.add_message("assistant", response))
        
    def load_conversation(self, conversation_id: str) -> None:
        """Load a conversation into the chat area."""
        self.logger.debug(f"Loading conversation: {conversation_id}")
        
        self.current_conversation_id = conversation_id
        
        # Update header
        self.title_label.setText(f"Conversation {conversation_id}")
        self.info_label.setText("Ready to chat")
        
        # Clear existing messages
        self.clear_messages()
        
        # Initialize RAG context for this conversation
        if self.rag_integration_service:
            try:
                # Create chat context for this conversation
                asyncio.create_task(
                    self.rag_integration_service.create_chat_context(conversation_id)
                )
            except Exception as e:
                self.logger.warning(f"Failed to create RAG context: {e}")
        
        # Load conversation messages (placeholder for now)
        self._load_sample_messages()
        
        # Focus input area
        self.input_area.set_focus()
        
    def _load_sample_messages(self) -> None:
        """Load sample messages for testing."""
        if self.current_conversation_id == "conv_1":
            self.add_message("user", "How do I create a virtual environment in Python?")
            self.add_message("assistant", "To create a virtual environment in Python, you can use the `venv` module. Here's how:\n\n1. Open your terminal\n2. Navigate to your project directory\n3. Run: `python -m venv myenv`\n4. Activate it with `myenv\\Scripts\\activate` (Windows) or `source myenv/bin/activate` (Mac/Linux)")
        elif self.current_conversation_id == "conv_2":
            self.add_message("user", "Can you explain neural networks?")
            self.add_message("assistant", "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process and transmit information. Each connection has a weight that adjusts during learning to improve the network's ability to recognize patterns and make predictions.")
            
    def add_message(self, role: str, content: str, timestamp: Optional[str] = None, evaluation_data: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the chat area."""
        message_bubble = MessageBubble(role, content, timestamp, evaluation_data)
        
        # Connect evaluation details signal
        if role == "assistant" and evaluation_data:
            message_bubble.evaluation_details_requested.connect(self._handle_evaluation_details_request)
            
        self.messages.append(message_bubble)
        self.chat_scroll.add_message_widget(message_bubble)
        
        # Emit evaluation signal if this is an assistant message with evaluation
        if role == "assistant" and evaluation_data:
            self.evaluation_received.emit(evaluation_data)
        
        self.logger.debug(f"Added {role} message: {content[:50]}...")
        
    def clear_messages(self) -> None:
        """Clear all messages from the chat area."""
        self.chat_scroll.clear_messages()
        self.messages.clear()
        self.current_streaming_message = None
        
    def get_conversation_id(self) -> Optional[str]:
        """Get the current conversation ID."""
        return self.current_conversation_id
        
    def set_loading(self, loading: bool) -> None:
        """Set loading state for the chat area."""
        self.input_area.setEnabled(not loading)
        if loading:
            self.info_label.setText("AI is thinking...")
        else:
            self.info_label.setText("Ready to chat")
            
    def set_rag_integration_service(self, rag_integration_service) -> None:
        """Set the RAG integration service."""
        self.rag_integration_service = rag_integration_service
        self._setup_rag_worker()
        self.logger.info("RAG integration service connected to chat area")
        
    def set_evaluation_service(self, evaluation_service: RAGEvaluationService) -> None:
        """Set the evaluation service."""
        self.evaluation_service = evaluation_service
        self.logger.info("Evaluation service connected to chat area")
        
    def _handle_evaluation_details_request(self, evaluation_data: Dict[str, Any]) -> None:
        """Handle request to show evaluation details."""
        try:
            # Create a detailed evaluation info string
            details = f"""
Evaluation Details:

Overall Score: {evaluation_data.get('overall_score', 0):.3f}

Metric Breakdown:
• Relevance: {evaluation_data.get('relevance_score', 0):.3f}
• Coherence: {evaluation_data.get('coherence_score', 0):.3f}
• Completeness: {evaluation_data.get('completeness_score', 0):.3f}
• Citation Accuracy: {evaluation_data.get('citation_accuracy', 0):.3f}
• Factual Accuracy: {evaluation_data.get('factual_accuracy', 0):.3f}

Performance:
• Response Time: {evaluation_data.get('response_time', 0):.2f}s
• Sources Used: {evaluation_data.get('retrieval_context', {}).get('total_retrieved', 0)}
• Confidence: {evaluation_data.get('confidence_score', 0):.3f}
"""
            
            # For now, log the details. In a full implementation,
            # this could open a detailed evaluation dialog
            self.logger.info(f"Evaluation details requested:\n{details}")
            
            # Update the status to show evaluation details are available
            self.rag_status_label.setText("RAG: Evaluation details available")
            self.rag_status_label.setStyleSheet("color: #00aaff;")  # Blue when showing details
            
        except Exception as e:
            self.logger.error(f"Failed to handle evaluation details request: {e}")
            
    def update_message_evaluation(self, message_index: int, evaluation_data: Dict[str, Any]) -> None:
        """Update evaluation data for a specific message."""
        try:
            if 0 <= message_index < len(self.messages):
                message = self.messages[message_index]
                if message.role == "assistant":
                    message.update_evaluation(evaluation_data)
                    self.evaluation_received.emit(evaluation_data)
                    self.logger.debug(f"Updated evaluation for message {message_index}")
        except Exception as e:
            self.logger.error(f"Failed to update message evaluation: {e}")
        
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.rag_thread and self.rag_thread.isRunning():
            self.rag_thread.quit()
            self.rag_thread.wait()
        self.logger.info("Chat area cleaned up")

    def resizeEvent(self, event):
        """Resize the drop overlay."""
        self.drop_overlay.resize(event.size())
        event.accept()

    def dragEnterEvent(self, event):
        """Handle drag enter events."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_overlay.show()

    def dragLeaveEvent(self, event):
        """Handle drag leave events."""
        self.drop_overlay.hide()

    def dropEvent(self, event):
        """Handle drop events for files."""
        self.drop_overlay.hide()
        urls = event.mimeData().urls()
        for url in urls:
            if url.isLocalFile():
                file_path = url.toLocalFile()
                self.logger.info(f"File dropped: {file_path}")
                # In a real implementation, you would process the file here
                # and add it to the message or a message bubble.
                self.add_message("user", f"Dropped file: {file_path}")


class DropOverlay(QWidget):
    """Overlay for showing drag-and-drop feedback."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, False)
        self.setAutoFillBackground(True)
        self.setStyleSheet("""
            background-color: rgba(0, 120, 212, 180);
            border-radius: 12px;
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        self.drop_label = QLabel("Drop Files Here")
        self.drop_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.drop_label.setStyleSheet("color: white;")
        
        layout.addWidget(self.drop_label)
        self.hide()