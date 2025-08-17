"""
Service for real-time collaboration and multi-modal content sharing.
"""

import asyncio
from typing import Any, Callable, Dict, List


class CollaborationService:
    """
    (Simulation) Manages real-time collaboration sessions and data sharing.
    """

    def __init__(self):
        self._sessions: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._is_connected = False

    async def connect(self, session_id: str):
        """Simulates connecting to a collaboration session."""
        if not self._is_connected:
            print(f"Connecting to collaboration session: {session_id}...")
            await asyncio.sleep(1)  # Simulate connection time
            self._is_connected = True
            self._sessions[session_id] = []
            print("Connected.")

    async def disconnect(self, session_id: str):
        """Simulates disconnecting from a collaboration session."""
        if self._is_connected:
            print(f"Disconnecting from session: {session_id}...")
            await asyncio.sleep(0.5)
            self._is_connected = False
            if session_id in self._sessions:
                del self._sessions[session_id]
            print("Disconnected.")

    async def share_content(self, session_id: str, content: Dict[str, Any]):
        """
        (Simulation) Shares content with all participants in a session.
        """
        if not self._is_connected or session_id not in self._sessions:
            print("Error: Not connected to a session.")
            return

        print(f"Sharing content in session {session_id}: {content}")
        await asyncio.sleep(0.5)  # Simulate network latency

        for callback in self._sessions.get(session_id, []):
            callback(content)

    def on_content_received(self, session_id: str, callback: Callable[[Dict[str, Any]], None]):
        """Registers a callback to be invoked when content is received."""
        if session_id in self._sessions:
            self._sessions[session_id].append(callback)


async def main():
    """Example usage of the CollaborationService."""
    collaboration_service = CollaborationService()

    def receiver(content: Dict[str, Any]):
        print(f"  - Client received content: {content}")

    session_id = "test_session_123"
    await collaboration_service.connect(session_id)
    collaboration_service.on_content_received(session_id, receiver)
    
    # Share some content
    await collaboration_service.share_content(session_id, {"type": "text", "data": "Hello, everyone!"})
    await collaboration_service.share_content(session_id, {"type": "image", "data": "path/to/image.jpg"})
    
    await collaboration_service.disconnect(session_id)

if __name__ == "__main__":
    asyncio.run(main())