"""
FastAPI server for voice agent.

Provides endpoints:
- POST /twiml - Returns TwiML to start Media Stream
- WS /ws - WebSocket endpoint for Twilio Media Streams
- GET /health - Health check
"""

import os
import logging

from fastapi import FastAPI, WebSocket, Response
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

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
app = FastAPI(title="Voice Agent")


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
    
    Delegates to the main event loop which handles:
    - VAD for speech detection
    - STT for transcription
    - LLM for response generation
    - TTS for speech synthesis
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    try:
        # Run the main event loop
        await run_call(websocket)
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    finally:
        logger.info("WebSocket connection closed")
