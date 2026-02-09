"""
FastAPI server for shuo.

Endpoints:
- GET /health - Health check
- GET/POST /twiml - Returns TwiML for Twilio to connect WebSocket
- WebSocket /ws - Media stream endpoint
- GET /trace/latest - Returns the most recent call trace as JSON
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from .conversation import run_conversation_over_twilio
from .log import get_logger

logger = get_logger("shuo.server")

app = FastAPI(title="shuo", docs_url=None, redoc_url=None)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.api_route("/twiml", methods=["GET", "POST"])
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
        <Stream url="{ws_url}" track="inbound_track" />
    </Connect>
</Response>"""
    
    return Response(content=twiml_response, media_type="application/xml")


@app.get("/trace/latest")
async def latest_trace():
    """Return the most recent call trace as JSON."""
    trace_dir = Path("/tmp/shuo")
    if not trace_dir.exists():
        return JSONResponse({"error": "No traces found"}, status_code=404)

    traces = sorted(trace_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not traces:
        return JSONResponse({"error": "No traces found"}, status_code=404)

    data = json.loads(traces[0].read_text())
    return JSONResponse(data)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    
    Handles the bidirectional audio stream for a single call.
    """
    await websocket.accept()
    
    try:
        await run_conversation_over_twilio(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
