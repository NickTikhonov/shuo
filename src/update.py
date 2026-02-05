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
    # State
    AppState, Phase, VADState, Message,
    # Events
    Event, StreamStartEvent, StreamStopEvent, MediaEvent,
    STTPartialEvent, STTFinalEvent,
    LLMTokenEvent, LLMDoneEvent,
    TTSAudioEvent, TTSDoneEvent,
    PlaybackDoneEvent,
    # Actions
    Action,
    StartSTTAction, FeedSTTAction, StopSTTAction, CancelSTTAction,
    StartLLMAction, CancelLLMAction,
    StartTTSAction, FeedTTSAction, FlushTTSAction, CancelTTSAction,
    StartPlaybackAction, StopPlaybackAction,
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
    # --- Twilio Events ---
    if isinstance(event, StreamStartEvent):
        return handle_stream_start(state, event.stream_sid)
    
    elif isinstance(event, StreamStopEvent):
        return handle_stream_stop(state)
    
    elif isinstance(event, MediaEvent):
        return handle_media(state, event.audio_bytes)
    
    # --- STT Events ---
    elif isinstance(event, STTPartialEvent):
        return handle_stt_partial(state, event.text)
    
    elif isinstance(event, STTFinalEvent):
        return handle_stt_final(state, event.text)
    
    # --- LLM Events ---
    elif isinstance(event, LLMTokenEvent):
        return handle_llm_token(state, event.token)
    
    elif isinstance(event, LLMDoneEvent):
        return handle_llm_done(state)
    
    # --- TTS Events ---
    elif isinstance(event, TTSAudioEvent):
        return handle_tts_audio(state, event.audio_base64)
    
    elif isinstance(event, TTSDoneEvent):
        return handle_tts_done(state)
    
    # --- Playback Events ---
    elif isinstance(event, PlaybackDoneEvent):
        return handle_playback_done(state)
    
    return state, []


# =============================================================================
# TWILIO EVENT HANDLERS
# =============================================================================

def handle_stream_start(state: AppState, stream_sid: str) -> Tuple[AppState, List[Action]]:
    """Handle new stream - reset everything."""
    logger.info(f"Stream started: {stream_sid}")
    reset_model_state()
    
    new_state = replace(
        state,
        phase=Phase.LISTENING,
        vad=VADState(),
        stream_sid=stream_sid,
        stt_active=False,
        current_transcript="",
        llm_active=False,
        pending_llm_text="",
        tts_active=False,
        # Keep conversation history
    )
    return new_state, []


def handle_stream_stop(state: AppState) -> Tuple[AppState, List[Action]]:
    """Handle stream end - cancel everything."""
    logger.info("Stream stopped")
    actions = []
    
    if state.stt_active:
        actions.append(CancelSTTAction())
    if state.llm_active:
        actions.append(CancelLLMAction())
    if state.tts_active:
        actions.append(CancelTTSAction())
    if state.phase == Phase.SPEAKING:
        actions.append(StopPlaybackAction())
    
    return state, actions


def handle_media(state: AppState, audio_bytes: bytes) -> Tuple[AppState, List[Action]]:
    """
    Handle incoming audio - run VAD and route based on phase.
    
    This is where the main logic happens:
    - LISTENING: Detect speech start/end, manage STT
    - PROCESSING: Detect interrupts
    - SPEAKING: Detect interrupts
    """
    # Decode and resample audio for VAD
    pcm = decode_mulaw(audio_bytes)
    pcm_16k = upsample_for_vad(pcm)
    
    # Run VAD
    new_vad, speech_started, speech_ended = process_audio(state.vad, pcm_16k)
    
    # Update VAD state
    state = replace(state, vad=new_vad)
    
    # Route based on phase
    if state.phase == Phase.LISTENING:
        return handle_media_listening(state, audio_bytes, speech_started, speech_ended)
    
    elif state.phase == Phase.PROCESSING:
        return handle_media_processing(state, audio_bytes, speech_started)
    
    elif state.phase == Phase.SPEAKING:
        return handle_media_speaking(state, audio_bytes, speech_started)
    
    return state, []


def handle_media_listening(
    state: AppState,
    audio_bytes: bytes,
    speech_started: bool,
    speech_ended: bool,
) -> Tuple[AppState, List[Action]]:
    """Handle audio in LISTENING phase."""
    actions: List[Action] = []
    
    # Speech just started - begin STT
    if speech_started:
        logger.info("User started speaking, starting STT")
        state = replace(state, stt_active=True, current_transcript="")
        actions.append(StartSTTAction())
    
    # If STT is active, feed it audio
    if state.stt_active:
        actions.append(FeedSTTAction(audio_bytes=audio_bytes))
    
    # Speech ended - stop STT and transition to PROCESSING
    if speech_ended and state.stt_active:
        logger.info("User finished speaking, stopping STT")
        actions.append(StopSTTAction())
        # Note: We don't transition yet - wait for STTFinalEvent
    
    return state, actions


def handle_media_processing(
    state: AppState,
    audio_bytes: bytes,
    speech_started: bool,
) -> Tuple[AppState, List[Action]]:
    """Handle audio in PROCESSING phase - check for interrupts."""
    if speech_started:
        return handle_interrupt(state)
    return state, []


def handle_media_speaking(
    state: AppState,
    audio_bytes: bytes,
    speech_started: bool,
) -> Tuple[AppState, List[Action]]:
    """Handle audio in SPEAKING phase - check for interrupts."""
    if speech_started:
        return handle_interrupt(state)
    return state, []


def handle_interrupt(state: AppState) -> Tuple[AppState, List[Action]]:
    """
    User interrupted - cancel everything and go back to listening.
    
    This is the critical path for responsiveness.
    """
    logger.info("User interrupted, cancelling all services")
    
    actions: List[Action] = []
    
    # Cancel any active services
    if state.llm_active:
        actions.append(CancelLLMAction())
    if state.tts_active:
        actions.append(CancelTTSAction())
    
    # Stop playback and clear buffer
    actions.append(StopPlaybackAction())
    
    # Start fresh STT for new utterance
    actions.append(StartSTTAction())
    
    # Reset state
    new_state = replace(
        state,
        phase=Phase.LISTENING,
        stt_active=True,
        current_transcript="",
        llm_active=False,
        pending_llm_text="",
        tts_active=False,
    )
    
    return new_state, actions


# =============================================================================
# STT EVENT HANDLERS
# =============================================================================

def handle_stt_partial(state: AppState, text: str) -> Tuple[AppState, List[Action]]:
    """Handle interim transcription - just update state for debugging."""
    # Could show partial transcript for debugging
    logger.debug(f"STT partial: {text}")
    return state, []


def handle_stt_final(state: AppState, text: str) -> Tuple[AppState, List[Action]]:
    """
    Handle final transcription - start LLM generation.
    
    This is the transition from LISTENING to PROCESSING.
    """
    logger.info(f"STT final: {text}")
    
    if not text.strip():
        logger.warning("Empty transcription, ignoring")
        return replace(state, stt_active=False), []
    
    # Build conversation history for LLM
    history = state.conversation_history
    
    # Transition to PROCESSING and start LLM
    new_state = replace(
        state,
        phase=Phase.PROCESSING,
        stt_active=False,
        current_transcript=text,
        llm_active=True,
        pending_llm_text="",
        tts_active=True,  # Start TTS immediately
    )
    
    actions = [
        StartLLMAction(history=history, user_message=text),
        StartTTSAction(),
    ]
    
    return new_state, actions


# =============================================================================
# LLM EVENT HANDLERS
# =============================================================================

def handle_llm_token(state: AppState, token: str) -> Tuple[AppState, List[Action]]:
    """
    Handle streamed token - forward to TTS.
    
    Tokens are sent to TTS immediately for lowest latency.
    """
    if not state.tts_active:
        return state, []
    
    # Accumulate for history
    new_state = replace(
        state,
        pending_llm_text=state.pending_llm_text + token
    )
    
    # Forward to TTS
    return new_state, [FeedTTSAction(text=token)]


def handle_llm_done(state: AppState) -> Tuple[AppState, List[Action]]:
    """
    Handle LLM completion - flush TTS.
    
    Tell TTS to synthesize any remaining buffered text.
    """
    logger.info("LLM generation complete")
    
    actions: List[Action] = []
    
    # Flush TTS to generate remaining audio
    if state.tts_active:
        actions.append(FlushTTSAction())
    
    # Update conversation history with assistant response
    new_history = state.conversation_history + (
        Message(role="user", content=state.current_transcript),
        Message(role="assistant", content=state.pending_llm_text),
    )
    
    new_state = replace(
        state,
        llm_active=False,
        conversation_history=new_history,
    )
    
    return new_state, actions


# =============================================================================
# TTS EVENT HANDLERS
# =============================================================================

def handle_tts_audio(state: AppState, audio_base64: str) -> Tuple[AppState, List[Action]]:
    """
    Handle TTS audio chunk - send to player.
    
    On first audio, transition to SPEAKING and start playback.
    """
    actions: List[Action] = []
    new_state = state
    
    # First audio chunk - transition to SPEAKING
    if state.phase == Phase.PROCESSING:
        logger.info("First TTS audio, starting playback")
        new_state = replace(state, phase=Phase.SPEAKING)
        actions.append(StartPlaybackAction())
    
    # Audio will be handled by the player via the effects layer
    # The player receives audio directly from TTS service callbacks
    
    return new_state, actions


def handle_tts_done(state: AppState) -> Tuple[AppState, List[Action]]:
    """Handle TTS completion."""
    logger.info("TTS synthesis complete")
    
    new_state = replace(state, tts_active=False)
    
    # If LLM is also done, we're just waiting for playback to finish
    # PlaybackDoneEvent will handle the transition back to LISTENING
    
    return new_state, []


# =============================================================================
# PLAYBACK EVENT HANDLERS
# =============================================================================

def handle_playback_done(state: AppState) -> Tuple[AppState, List[Action]]:
    """
    Handle playback completion - return to listening.
    
    Only transition if we're still in SPEAKING phase (not interrupted).
    """
    if state.phase != Phase.SPEAKING:
        return state, []
    
    logger.info("Playback complete, returning to listening")
    
    new_state = replace(
        state,
        phase=Phase.LISTENING,
    )
    
    return new_state, []
