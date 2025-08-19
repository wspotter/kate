"""
Standalone FastAPI Web Server for Kate LLM Client
This version runs independently without requiring the full Kate application.
"""
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn


class ChatMessage(BaseModel):
    content: str
    role: str = "user"


class ChatResponse(BaseModel):
    content: str
    role: str = "assistant"


class StandaloneWebServer:
    def __init__(self):
        self.app = FastAPI(title="Kate LLM Client", version="1.0.0")
        
        # Setup templates and static files
        self.templates = Jinja2Templates(directory="app/templates")
        
        # Create static directory if it doesn't exist
        Path("app/static").mkdir(exist_ok=True)
        
        # Mount static files
        try:
            self.app.mount("/static", StaticFiles(directory="app/static"), name="static")
        except:
            # Create empty static directory if it fails
            pass
        
        # Active WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup all web routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})
        
        @self.app.get("/chat", response_class=HTMLResponse)
        async def chat_page(request: Request):
            return self.templates.TemplateResponse("chat.html", {"request": request})
        
        @self.app.post("/api/chat")
        async def chat_endpoint(message: ChatMessage) -> ChatResponse:
            """Handle chat messages via REST API"""
            try:
                # Simple echo response for now - will integrate with RAG later
                response_content = f"Echo: {message.content}"
                
                # Simulate some processing delay
                await asyncio.sleep(0.5)
                
                return ChatResponse(
                    content=response_content,
                    role="assistant"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy", 
                "message": "Kate Web Server is running!",
                "services": {
                    "web_server": True,
                    "templates": Path("app/templates").exists(),
                    "static": Path("app/static").exists()
                }
            }
        
        @self.app.websocket("/ws/chat")
        async def websocket_chat(websocket: WebSocket):
            """WebSocket endpoint for real-time chat"""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Receive message from client
                    data = await websocket.receive_text()
                    
                    # Simple echo response
                    response = f"WebSocket Echo: {data}"
                    
                    # Send response back
                    await websocket.send_text(response)
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
        
        @self.app.get("/api/settings")
        async def get_settings():
            """Get current Kate settings"""
            return {
                "voice_enabled": True,
                "rag_enabled": True,
                "models_available": ["gpt-3.5-turbo", "claude-3", "local-llama"],
                "web_server_mode": "standalone"
            }


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    web_server = StandaloneWebServer()
    return web_server.app


async def run_web_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the web server"""
    app = create_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    print(f"🚀 Kate Web Server starting at http://{host}:{port}")
    print(f"📱 Open your browser to http://{host}:{port} to use Kate!")
    
    await server.serve()


if __name__ == "__main__":
    asyncio.run(run_web_server())