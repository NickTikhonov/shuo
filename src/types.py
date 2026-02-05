"""
Type definitions for the voice agent.

All state, events, and actions are immutable dataclasses.
This makes the system predictable and easy to test.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Union


# =============================================================================
# STATE
# =============================================================================

class Phase(Enum):
    """Current phase of the conversation."""
    LISTENING = auto()  # Waiting for user to finish speaking
    PLAYING = auto()    # Playing response audio


@dataclass(frozen=True)
class VADState:
    """Voice Activity Detection state."""
    is_speaking: bool = False
    speech_start_time: Optional[float] = None
    silence_start_time: Optional[float] = None
    audio_buffer: Tuple[float, ...] = ()


@dataclass(frozen=True)
class AppState:
    """Complete application state - single source of truth."""
    phase: Phase = Phase.LISTENING
    vad: VADState = field(default_factory=VADState)
    stream_sid: Optional[str] = None
    response_chunks: Tuple[str, ...] = ()  # Pre-loaded audio (base64 mulaw)


# =============================================================================
# EVENTS (things that happen - inputs to the system)
# =============================================================================

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


# Union of all event types
Event = Union[StreamStartEvent, StreamStopEvent, MediaEvent]


# =============================================================================
# ACTIONS (side effects to perform - outputs from the system)
# =============================================================================

@dataclass(frozen=True)
class StartPlaybackAction:
    """Start playing audio (all chunks)."""
    chunks: Tuple[str, ...]


@dataclass(frozen=True)
class StopPlaybackAction:
    """Stop playback and clear Twilio buffer."""
    pass


# Union of all action types
Action = Union[StartPlaybackAction, StopPlaybackAction]
