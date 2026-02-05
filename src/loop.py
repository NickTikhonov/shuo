"""
The main event loop for shuo.

This is the explicit, readable loop that drives the entire system:

    while connected:
        event = receive()                      # I/O (from queue)
        state, actions = update(state, event)  # PURE
        for action in actions:
            execute(action)                    # I/O

Events come from multiple sources:
- Twilio WebSocket (audio packets)
- STT Service (transcriptions)
- LLM Service (tokens)
- TTS Service (audio chunks)
- Player (playback complete)

All sources push to a shared async queue, which the main loop consumes.
"""

import json
import base64
import asyncio
from typing import Optional

from fastapi import WebSocket

from .types import (
    AppState, Phase, VADState,
    Event, StreamStartEvent, StreamStopEvent, MediaEvent,
    STTPartialEvent, STTFinalEvent,
    LLMTokenEvent, LLMDoneEvent,
    TTSAudioEvent, TTSDoneEvent,
    PlaybackDoneEvent,
)
from .update import update
from .effects import EffectsExecutor
from .player import AudioPlayer
from .services.stt import STTService
from .services.llm import LLMService
from .services.tts import TTSService
from .log import Lifecycle, EventLogger, get_logger

logger = get_logger("shuo.loop")


def parse_twilio_message(data: dict) -> Optional[Event]:
    """
    Parse raw Twilio WebSocket message into typed Event.
    
    Returns None for events we don't care about.
    """
    event_type = data.get("event")
    
    if event_type == "connected":
        Lifecycle.websocket_connected()
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


async def run_call(websocket: WebSocket) -> None:
    """
    Main event loop for a single call.
    
    This coordinates all services and manages the event-driven architecture:
    1. Create shared event queue
    2. Create services with callbacks that push to queue
    3. Start Twilio reader task
    4. Process events through pure update function
    5. Execute actions through effects executor
    """
    # Event logger for this call
    event_log = EventLogger(verbose=False)
    
    # Shared event queue - all sources push here
    event_queue: asyncio.Queue[Event] = asyncio.Queue()
    
    # Will be initialized when we get stream_sid
    player: Optional[AudioPlayer] = None
    executor: Optional[EffectsExecutor] = None
    
    # --- Service Callbacks (push events to queue) ---
    
    async def on_stt_partial(text: str) -> None:
        await event_queue.put(STTPartialEvent(text=text))
    
    async def on_stt_final(text: str) -> None:
        await event_queue.put(STTFinalEvent(text=text))
    
    async def on_llm_token(token: str) -> None:
        await event_queue.put(LLMTokenEvent(token=token))
    
    async def on_llm_done() -> None:
        await event_queue.put(LLMDoneEvent())
    
    async def on_tts_audio(audio_base64: str) -> None:
        await event_queue.put(TTSAudioEvent(audio_base64=audio_base64))
        # Also send audio to player directly
        if player:
            await player.send_chunk(audio_base64)
    
    async def on_tts_done() -> None:
        await event_queue.put(TTSDoneEvent())
        # Mark TTS done so player knows no more chunks coming
        if player:
            player.mark_tts_done()
    
    # --- Create Services ---
    
    stt = STTService(on_partial=on_stt_partial, on_final=on_stt_final)
    llm = LLMService(on_token=on_llm_token, on_done=on_llm_done)
    tts = TTSService(on_audio=on_tts_audio, on_done=on_tts_done)
    
    # --- Twilio WebSocket Reader Task ---
    
    async def read_twilio() -> None:
        """Background task to read from Twilio and push to event queue."""
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                event = parse_twilio_message(data)
                if event:
                    await event_queue.put(event)
                    if isinstance(event, StreamStopEvent):
                        break
        except Exception as e:
            event_log.error("Twilio reader", e)
            await event_queue.put(StreamStopEvent())
    
    # --- Initialize State ---
    
    state = AppState(
        phase=Phase.LISTENING,
        vad=VADState(),
        stream_sid=None,
    )
    
    # Start Twilio reader
    reader_task = asyncio.create_task(read_twilio())
    
    try:
        while True:
            # ─────────────────────────────────────────────────────────────
            # RECEIVE: Wait for event from any source
            # ─────────────────────────────────────────────────────────────
            event = await event_queue.get()
            
            # Log lifecycle events
            if isinstance(event, StreamStartEvent):
                Lifecycle.stream_started(event.stream_sid)
            elif isinstance(event, StreamStopEvent):
                Lifecycle.stream_stopped()
            
            # Log the event
            event_log.event(event)
            
            # Initialize player and executor when we get stream_sid
            if isinstance(event, StreamStartEvent) and player is None:
                player = AudioPlayer(
                    websocket=websocket,
                    stream_sid=event.stream_sid,
                    on_done=lambda: event_queue.put_nowait(PlaybackDoneEvent()),
                )
                executor = EffectsExecutor(
                    player=player,
                    stt=stt,
                    llm=llm,
                    tts=tts,
                )
            
            # ─────────────────────────────────────────────────────────────
            # UPDATE: Pure state transition
            # ─────────────────────────────────────────────────────────────
            old_phase = state.phase
            state, actions = update(state, event)
            
            # Log phase transition
            event_log.transition(old_phase, state.phase)
            
            # ─────────────────────────────────────────────────────────────
            # EXECUTE: Perform side effects
            # ─────────────────────────────────────────────────────────────
            if executor:
                for action in actions:
                    event_log.action(action)
                    await executor.execute(action)
            
            # ─────────────────────────────────────────────────────────────
            # CHECK: Exit condition
            # ─────────────────────────────────────────────────────────────
            if isinstance(event, StreamStopEvent):
                break
                
    except Exception as e:
        event_log.error("Call loop", e)
        raise
    
    finally:
        # Cleanup
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass
        
        # Cancel any active services
        await stt.cancel()
        await llm.cancel()
        await tts.cancel()
        
        if player and player.is_playing:
            await player.stop_and_clear()
        
        Lifecycle.websocket_disconnected()
