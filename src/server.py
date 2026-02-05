"""
FastAPI server for voice agent.

Provides endpoints:
- POST /twiml - Returns TwiML to start Media Stream
- WS /ws - WebSocket endpoint for Twilio Media Streams
- GET /health - Health check
"""

import os
import logging
from typing import Tuple

from fastapi import FastAPI, WebSocket, Response
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

from .audio import load_response_audio
from .loop import run_call

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

# Pre-loaded response audio chunks (immutable tuple)
_response_chunks: Tuple[str, ...] = ()


def init_response_audio(audio_path: str) -> None:
    """Load response audio at startup."""
    global _response_chunks
    logger.info(f"Loading response audio from {audio_path}")
    chunks = load_response_audio(audio_path)
    _response_chunks = tuple(chunks)
    logger.info(f"Loaded {len(_response_chunks)} audio chunks")


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
    
    Delegates to the main event loop.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    if not _response_chunks:
        logger.error("Response audio not loaded")
        await websocket.close()
        return
    
    try:
        # Run the main event loop
        await run_call(websocket, _response_chunks)
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    finally:
        logger.info("WebSocket connection closed")
