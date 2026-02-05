"""
FastAPI server for voice agent.

Provides endpoints:
- GET /twiml - Returns TwiML to start Media Stream
- WS /ws - WebSocket endpoint for Twilio Media Streams
- GET /health - Health check
"""

import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, Response
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

from .websocket_handler import TwilioMediaStreamHandler
from .conversation import ConversationManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Voice Agent VAD System")

# Global conversation manager (will be initialized with response audio)
conversation_manager: Optional[ConversationManager] = None


def init_conversation_manager(response_audio_path: str) -> None:
    """Initialize the global conversation manager with response audio."""
    global conversation_manager
    conversation_manager = ConversationManager()
    conversation_manager.load_response(response_audio_path)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/twiml", response_class=PlainTextResponse)
async def get_twiml():
    """
    Return TwiML that starts a bidirectional Media Stream.
    
    Twilio will call this URL when the outbound call is answered,
    then connect to our WebSocket endpoint.
    """
    public_url = os.getenv("TWILIO_PUBLIC_URL", "")
    
    # Convert https:// to wss:// for WebSocket
    ws_url = public_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/ws"
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="direction" value="both"/>
        </Stream>
    </Connect>
</Response>"""
    
    logger.info(f"Returning TwiML with WebSocket URL: {ws_url}")
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    
    Handles bidirectional audio streaming between Twilio and our VAD system.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    if conversation_manager is None:
        logger.error("Conversation manager not initialized")
        await websocket.close()
        return
    
    # Create handler with a fresh conversation state
    # Note: We create a new ConversationManager for each call
    # but share the loaded response audio
    call_conversation = ConversationManager()
    call_conversation.response_chunks = conversation_manager.response_chunks
    
    handler = TwilioMediaStreamHandler(
        websocket=websocket,
        conversation=call_conversation
    )
    
    try:
        await handler.handle()
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    finally:
        logger.info("WebSocket connection closed")
