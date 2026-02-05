"""
FastAPI server for shuo.

Endpoints:
- GET /health - Health check
- POST /twiml - Returns TwiML for Twilio to connect WebSocket
- WebSocket /ws - Media stream endpoint
"""

import os
from fastapi import FastAPI, WebSocket, Response
from fastapi.responses import PlainTextResponse

from .loop import run_call
from .log import get_logger

logger = get_logger("shuo.server")

app = FastAPI(title="shuo", docs_url=None, redoc_url=None)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/twiml")
async def twiml():
    """
    Return TwiML instructing Twilio to connect a WebSocket stream.
    
    Twilio calls this URL when the call is answered.
    """
    public_url = os.getenv("TWILIO_PUBLIC_URL", "")
    ws_url = public_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/ws"
    
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>"""
    
    return Response(content=twiml_response, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    
    Handles the bidirectional audio stream for a single call.
    """
    await websocket.accept()
    
    try:
        await run_call(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
