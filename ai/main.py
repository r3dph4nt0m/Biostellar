from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import logging
import os
from dotenv import load_dotenv
from typing import List
import datetime

from langchain_cerebras import ChatCerebras
from langchain.schema import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="ChatCerebras WebSocket Chat", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_chat_history = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id not in self.user_chat_history:
            self.user_chat_history[user_id] = []
        logger.info(f"User {user_id} connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, user_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"User {user_id} disconnected. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

def get_cerebras_llm():
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        logger.error("CEREBRAS_API_KEY not set in environment.")
        raise RuntimeError("CEREBRAS_API_KEY not set")
    return ChatCerebras(
        model="llama-3.3-70b",
        api_key=api_key,
        temperature=0.7,
        max_tokens=512
    )

def get_ai_response(user_message: str, chat_history: List[str]) -> str:
    """Get AI response using Cerebras LLM with simple chat history"""
    try:
        llm = get_cerebras_llm()
        
        # Build context from chat history
        context = "You are a helpful AI assistant. Please respond to the user's messages in a friendly and informative way.\n\n"
        
        # Add recent chat history (last 5 exchanges)
        recent_history = chat_history[-10:]  # Last 10 messages (5 exchanges)
        for msg in recent_history:
            context += f"{msg}\n"
        
        # Add current user message
        context += f"User: {user_message}\nAssistant:"
        
        # Get response from LLM
        response = llm.invoke(context)
        return response.content if hasattr(response, 'content') else str(response)
        
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return f"I apologize, but I encountered an error: {str(e)}"

@app.get("/")
async def read_root():
    return {"message": "ChatCerebras WebSocket backend running."}

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received message from user {user_id}: {data}")
            
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            if not user_message:
                logger.warning(f"Empty message received from user {user_id}")
                continue
            
            chat_history = manager.user_chat_history.get(user_id, [])
            logger.debug(f"Current chat history for user {user_id}: {chat_history}")
            
            await manager.send_personal_message(
                json.dumps({"type": "typing", "message": "AI is thinking..."}),
                websocket
            )
            
            try:
                ai_response = get_ai_response(user_message, chat_history)
                logger.info(f"AI response generated for user {user_id}: {ai_response[:100]}...")
                
                chat_history.append(f"User: {user_message}")
                chat_history.append(f"Assistant: {ai_response}")
                
                if len(chat_history) > 20:
                    logger.debug(f"Trimming chat history for user {user_id}")
                    chat_history = chat_history[-20:]
                
                manager.user_chat_history[user_id] = chat_history
                
                timestamp = datetime.datetime.utcnow().isoformat()
                response_data = {
                    "type": "ai_message",
                    "message": ai_response,
                    "timestamp": timestamp
                }
                logger.debug(f"Sending response to user {user_id}: {response_data}")
                
                await manager.send_personal_message(
                    json.dumps(response_data),
                    websocket
                )
                
            except Exception as e:
                logger.error(f"Error processing message for user {user_id}: {str(e)}", exc_info=True)
                await manager.send_personal_message(
                    json.dumps({"error": f"Error processing message: {str(e)}"}),
                    websocket
                )
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}", exc_info=True)
        manager.disconnect(websocket, user_id)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "connections": len(manager.active_connections)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
