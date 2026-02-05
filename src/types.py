"""
Type definitions for the voice agent.

All state, events, and actions are immutable dataclasses.
This makes the system predictable and easy to test.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Union, List


# =============================================================================
# STATE
# =============================================================================

class Phase(Enum):
    """Current phase of the conversation."""
    LISTENING = auto()    # Waiting for user / user speaking / STT running
    PROCESSING = auto()   # STT complete, LLM generating, TTS may be generating
    SPEAKING = auto()     # TTS audio playing to user


@dataclass(frozen=True)
class VADState:
    """Voice Activity Detection state."""
    is_speaking: bool = False
    speech_start_time: Optional[float] = None
    silence_start_time: Optional[float] = None
    audio_buffer: Tuple[float, ...] = ()


@dataclass(frozen=True)
class Message:
    """A message in the conversation history."""
    role: str  # "user" or "assistant"
    content: str


@dataclass(frozen=True)
class AppState:
    """Complete application state - single source of truth."""
    phase: Phase = Phase.LISTENING
    vad: VADState = field(default_factory=VADState)
    stream_sid: Optional[str] = None
    
    # Conversation history for LLM context
    conversation_history: Tuple[Message, ...] = ()
    
    # Current turn state
    stt_active: bool = False
    current_transcript: str = ""
    
    # LLM state
    llm_active: bool = False
    pending_llm_text: str = ""  # Accumulated LLM output for current turn
    
    # TTS state
    tts_active: bool = False


# =============================================================================
# EVENTS (things that happen - inputs to the system)
# =============================================================================

# --- Twilio Events ---

@dataclass(frozen=True)
class StreamStartEvent:
    """Twilio stream started."""
    stream_sid: str


@dataclass(frozen=True)
class StreamStopEvent:
    """Twilio stream ended."""
    pass


@dataclass(frozen=True)
class MediaEvent:
    """Audio data received from Twilio."""
    audio_bytes: bytes


# --- STT Events ---

@dataclass(frozen=True)
class STTPartialEvent:
    """Interim transcription from Deepgram."""
    text: str


@dataclass(frozen=True)
class STTFinalEvent:
    """Final transcription from Deepgram."""
    text: str


# --- LLM Events ---

@dataclass(frozen=True)
class LLMTokenEvent:
    """Streamed token from OpenAI."""
    token: str


@dataclass(frozen=True)
class LLMDoneEvent:
    """LLM generation complete."""
    pass


# --- TTS Events ---

@dataclass(frozen=True)
class TTSAudioEvent:
    """Audio chunk from ElevenLabs (base64 encoded)."""
    audio_base64: str


@dataclass(frozen=True)
class TTSDoneEvent:
    """TTS synthesis complete."""
    pass


# --- Playback Events ---

@dataclass(frozen=True)
class PlaybackDoneEvent:
    """Audio playback finished."""
    pass


# Union of all event types
Event = Union[
    # Twilio
    StreamStartEvent, StreamStopEvent, MediaEvent,
    # STT
    STTPartialEvent, STTFinalEvent,
    # LLM
    LLMTokenEvent, LLMDoneEvent,
    # TTS
    TTSAudioEvent, TTSDoneEvent,
    # Playback
    PlaybackDoneEvent,
]


# =============================================================================
# ACTIONS (side effects to perform - outputs from the system)
# =============================================================================

# --- STT Actions ---

@dataclass(frozen=True)
class StartSTTAction:
    """Open Deepgram WebSocket connection."""
    pass


@dataclass(frozen=True)
class FeedSTTAction:
    """Send audio chunk to Deepgram."""
    audio_bytes: bytes


@dataclass(frozen=True)
class StopSTTAction:
    """Close Deepgram connection and get final transcript."""
    pass


@dataclass(frozen=True)
class CancelSTTAction:
    """Abort Deepgram connection without waiting."""
    pass


# --- LLM Actions ---

@dataclass(frozen=True)
class StartLLMAction:
    """Begin LLM generation with conversation history."""
    history: Tuple[Message, ...]
    user_message: str


@dataclass(frozen=True)
class CancelLLMAction:
    """Abort LLM generation."""
    pass


# --- TTS Actions ---

@dataclass(frozen=True)
class StartTTSAction:
    """Open ElevenLabs WebSocket connection."""
    pass


@dataclass(frozen=True)
class FeedTTSAction:
    """Send text chunk to ElevenLabs."""
    text: str


@dataclass(frozen=True)
class FlushTTSAction:
    """Force synthesis of buffered text."""
    pass


@dataclass(frozen=True)
class CancelTTSAction:
    """Abort TTS synthesis."""
    pass


# --- Playback Actions ---

@dataclass(frozen=True)
class StartPlaybackAction:
    """Start playing audio - player will receive chunks via TTSAudioEvent."""
    pass


@dataclass(frozen=True)
class StopPlaybackAction:
    """Stop playback and clear Twilio buffer."""
    pass


# Union of all action types
Action = Union[
    # STT
    StartSTTAction, FeedSTTAction, StopSTTAction, CancelSTTAction,
    # LLM
    StartLLMAction, CancelLLMAction,
    # TTS
    StartTTSAction, FeedTTSAction, FlushTTSAction, CancelTTSAction,
    # Playback
    StartPlaybackAction, StopPlaybackAction,
]
