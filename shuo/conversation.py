"""
The main event loop for shuo.

This is the explicit, readable loop that drives the entire system:

    while connected:
        event = receive()                               # I/O (from queue)
        state, actions = process_event(state, event)    # PURE
        for action in actions:
            dispatch(action)                            # I/O

Events come from:
- Twilio WebSocket (audio packets)
- Deepgram Flux (turn events)
- Agent (playback complete)
"""

import json
import asyncio
from typing import Optional

from fastapi import WebSocket

from .types import (
    AppState,
    Event, StreamStartEvent, StreamStopEvent,
    FluxStartOfTurnEvent, FluxEndOfTurnEvent, AgentTurnDoneEvent,
    FeedFluxAction, StartAgentTurnAction, ResetAgentTurnAction,
)
from .state import process_event
from .services.flux import FluxService
from .services.tts_pool import TTSPool
from .services.twilio_client import parse_twilio_message
from .agent import Agent
from .log import Logger, get_logger

logger = get_logger("shuo.conversation")


async def run_conversation_over_twilio(websocket: WebSocket) -> None:
    """
    Main event loop for a single call.

    1. Create shared event queue
    2. Create Flux service (always-on STT + turn detection)
    3. Start Twilio reader
    4. On StreamStart, create Agent
    5. Process events through pure state machine
    6. Dispatch actions inline
    """
    event_log = Logger(verbose=False)
    event_queue: asyncio.Queue[Event] = asyncio.Queue()

    agent: Optional[Agent] = None
    tts_pool = TTSPool(pool_size=1, ttl=8.0)

    # ── Flux Callbacks (push events to queue) ───────────────────────

    async def on_flux_end_of_turn(transcript: str) -> None:
        await event_queue.put(FluxEndOfTurnEvent(transcript=transcript))

    async def on_flux_start_of_turn() -> None:
        await event_queue.put(FluxStartOfTurnEvent())

    # ── Create Flux Service ─────────────────────────────────────────

    flux = FluxService(
        on_end_of_turn=on_flux_end_of_turn,
        on_start_of_turn=on_flux_start_of_turn,
    )

    # ── Twilio WebSocket Reader ─────────────────────────────────────

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

    # ── Initialize ──────────────────────────────────────────────────

    state = AppState()
    reader_task = asyncio.create_task(read_twilio())

    try:
        while True:
            # ─── RECEIVE ────────────────────────────────────────────
            event = await event_queue.get()

            event_log.event(event)

            # Initialize services on stream start
            if isinstance(event, StreamStartEvent):
                await flux.start()
                await tts_pool.start()
                agent = Agent(
                    websocket=websocket,
                    stream_sid=event.stream_sid,
                    on_done=lambda: event_queue.put_nowait(AgentTurnDoneEvent()),
                    tts_pool=tts_pool,
                )

            # ─── UPDATE (pure) ──────────────────────────────────────
            old_phase = state.phase
            state, actions = process_event(state, event)
            event_log.transition(old_phase, state.phase)

            # ─── DISPATCH (side effects) ────────────────────────────
            for action in actions:
                event_log.action(action)
                if isinstance(action, FeedFluxAction):
                    await flux.send(action.audio_bytes)

                elif isinstance(action, StartAgentTurnAction):
                    if agent:
                        await agent.start_turn(action.transcript)

                elif isinstance(action, ResetAgentTurnAction):
                    if agent:
                        await agent.cancel_turn()

            # ─── EXIT CHECK ─────────────────────────────────────────
            if isinstance(event, StreamStopEvent):
                break

    except Exception as e:
        event_log.error("Call loop", e)
        raise

    finally:
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass

        if agent:
            await agent.cleanup()

        await tts_pool.stop()
        await flux.stop()

        Logger.websocket_disconnected()
