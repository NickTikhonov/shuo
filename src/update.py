"""
Pure state machine for the voice agent.

The update function is the heart of the system:
    (State, Event) -> (State, List[Action])

No side effects, no I/O - just pure state transitions.
This makes the entire system testable and predictable.
"""

import logging
from dataclasses import replace
from typing import List, Tuple

from .types import (
    AppState, Phase, VADState,
    Event, StreamStartEvent, StreamStopEvent, MediaEvent,
    Action, StartPlaybackAction, StopPlaybackAction,
)
from .audio import decode_mulaw, upsample_for_vad
from .vad import process_audio, reset_model_state

logger = logging.getLogger(__name__)


def update(state: AppState, event: Event) -> Tuple[AppState, List[Action]]:
    """
    Pure state machine: (State, Event) -> (State, Actions)
    
    This is the ONLY place where state transitions happen.
    All logic is concentrated here, making it easy to understand and test.
    """
    if isinstance(event, StreamStartEvent):
        return handle_stream_start(state, event.stream_sid)
    
    elif isinstance(event, StreamStopEvent):
        return handle_stream_stop(state)
    
    elif isinstance(event, MediaEvent):
        return handle_media(state, event.audio_bytes)
    
    return state, []


def handle_stream_start(state: AppState, stream_sid: str) -> Tuple[AppState, List[Action]]:
    """Handle new stream - reset state."""
    logger.info(f"Stream started: {stream_sid}")
    reset_model_state()
    
    new_state = replace(
        state,
        phase=Phase.LISTENING,
        vad=VADState(),
        stream_sid=stream_sid,
    )
    return new_state, []


def handle_stream_stop(state: AppState) -> Tuple[AppState, List[Action]]:
    """Handle stream end."""
    logger.info("Stream stopped")
    actions = []
    if state.phase == Phase.PLAYING:
        actions.append(StopPlaybackAction())
    return state, actions


def handle_media(state: AppState, audio_bytes: bytes) -> Tuple[AppState, List[Action]]:
    """
    Handle incoming audio - the main logic.
    
    This is where VAD runs and state transitions happen.
    """
    # Decode and resample audio
    pcm = decode_mulaw(audio_bytes)
    pcm_16k = upsample_for_vad(pcm)
    
    # Run VAD
    new_vad, speech_started, speech_ended = process_audio(state.vad, pcm_16k)
    
    # Handle based on current phase
    if state.phase == Phase.LISTENING:
        return handle_listening_phase(state, new_vad, speech_started, speech_ended)
    
    elif state.phase == Phase.PLAYING:
        return handle_playing_phase(state, new_vad, speech_started)
    
    return replace(state, vad=new_vad), []


def handle_listening_phase(
    state: AppState,
    new_vad: VADState,
    speech_started: bool,
    speech_ended: bool,
) -> Tuple[AppState, List[Action]]:
    """Handle audio while in LISTENING phase."""
    
    if speech_started:
        logger.info("User started speaking")
    
    if speech_ended:
        logger.info("User finished speaking, starting playback")
        
        if not state.response_chunks:
            logger.warning("No response audio loaded")
            return replace(state, vad=new_vad), []
        
        # Transition to PLAYING
        new_state = replace(
            state,
            phase=Phase.PLAYING,
            vad=new_vad,
        )
        
        # Start playback with all chunks
        actions = [StartPlaybackAction(chunks=state.response_chunks)]
        
        return new_state, actions
    
    return replace(state, vad=new_vad), []


def handle_playing_phase(
    state: AppState,
    new_vad: VADState,
    speech_started: bool,
) -> Tuple[AppState, List[Action]]:
    """Handle audio while in PLAYING phase - check for interrupts."""
    
    if speech_started:
        logger.info("User interrupted, stopping playback")
        
        # Transition back to LISTENING
        new_state = replace(
            state,
            phase=Phase.LISTENING,
            vad=new_vad,
        )
        
        # Stop playback
        actions = [StopPlaybackAction()]
        
        return new_state, actions
    
    return replace(state, vad=new_vad), []
