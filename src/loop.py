"""
The main event loop for the voice agent.

This is the explicit, readable loop that drives the entire system:

    while connected:
        event = receive()                    # I/O
        state, actions = update(state, event)  # PURE
        for action in actions:
            execute(action)                  # I/O

Playback runs independently via AudioPlayer - no timing conflicts.
"""

import json
import base64
import logging
from typing import Optional, Tuple

from fastapi import WebSocket

from .types import (
    AppState, Phase, VADState,
    Event, StreamStartEvent, StreamStopEvent, MediaEvent,
)
from .update import update
from .effects import execute
from .player import AudioPlayer

logger = logging.getLogger(__name__)


def parse_twilio_message(data: dict) -> Optional[Event]:
    """
    Parse raw Twilio WebSocket message into typed Event.
    
    Returns None for events we don't care about.
    """
    event_type = data.get("event")
    
    if event_type == "connected":
        # Connection established, but we wait for "start" for stream_sid
        logger.info("Twilio connected")
        return None
    
    elif event_type == "start":
        start_data = data.get("start", {})
        stream_sid = start_data.get("streamSid")
        if stream_sid:
            return StreamStartEvent(stream_sid=stream_sid)
    
    elif event_type == "media":
        media_data = data.get("media", {})
        payload = media_data.get("payload", "")
        if payload:
            audio_bytes = base64.b64decode(payload)
            return MediaEvent(audio_bytes=audio_bytes)
    
    elif event_type == "stop":
        return StreamStopEvent()
    
    return None


async def run_call(websocket: WebSocket, response_chunks: Tuple[str, ...]) -> None:
    """
    Main event loop for a single call.
    
    This is the heart of the system - a simple, explicit loop:
    1. RECEIVE event from WebSocket
    2. UPDATE state (pure function)
    3. EXECUTE actions (side effects)
    
    Playback runs independently in AudioPlayer.
    """
    # Initialize state
    state = AppState(
        phase=Phase.LISTENING,
        vad=VADState(),
        stream_sid=None,
        response_chunks=response_chunks,
    )
    
    # Player will be created once we have stream_sid
    player: Optional[AudioPlayer] = None
    
    logger.info("Starting call loop")
    
    try:
        while True:
            # ─────────────────────────────────────────────────────────────
            # RECEIVE: Wait for WebSocket message
            # ─────────────────────────────────────────────────────────────
            raw = await websocket.receive_text()
            data = json.loads(raw)
            event = parse_twilio_message(data)
            
            if event is None:
                continue
            
            # Create player once we have stream_sid
            if isinstance(event, StreamStartEvent) and player is None:
                player = AudioPlayer(websocket, event.stream_sid)
            
            # ─────────────────────────────────────────────────────────────
            # UPDATE: Pure state transition
            # ─────────────────────────────────────────────────────────────
            state, actions = update(state, event)
            
            # ─────────────────────────────────────────────────────────────
            # EXECUTE: Perform side effects
            # ─────────────────────────────────────────────────────────────
            if player:
                for action in actions:
                    await execute(action, player)
            
            # ─────────────────────────────────────────────────────────────
            # CHECK: Exit condition
            # ─────────────────────────────────────────────────────────────
            if isinstance(event, StreamStopEvent):
                logger.info("Stream stopped, exiting loop")
                break
                
    except Exception as e:
        logger.error(f"Call loop error: {e}")
        raise
    finally:
        if player and player.is_playing:
            await player.stop_and_clear()
        logger.info("Call loop ended")
