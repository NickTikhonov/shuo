"""
Conversation state machine for voice agent.

Manages the flow between:
- LISTENING: Waiting for user to speak and finish
- PLAYING: Streaming response audio to user
- Handles interrupts by immediately stopping playback
"""

import asyncio
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional, List

from .vad import VoiceActivityDetector, SpeechState, VADConfig
from .audio import load_response_audio, upsample_for_vad

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Current state of the conversation."""
    LISTENING = auto()  # Waiting for user input
    PLAYING = auto()    # Playing response audio


@dataclass
class ConversationManager:
    """
    Manages conversation state and audio playback.
    
    Coordinates between VAD, audio streaming, and the WebSocket connection.
    """
    
    # Callbacks for sending audio and clearing the buffer
    send_audio: Optional[Callable[[str], Awaitable[None]]] = None
    send_clear: Optional[Callable[[], Awaitable[None]]] = None
    
    # Response audio chunks (base64 mulaw)
    response_chunks: List[str] = field(default_factory=list)
    
    # Current state
    state: ConversationState = ConversationState.LISTENING
    
    # VAD instance
    vad: VoiceActivityDetector = field(default_factory=VoiceActivityDetector)
    
    # Playback state
    _playback_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _chunk_index: int = 0
    
    # Stream ID from Twilio (needed for audio messages)
    stream_sid: Optional[str] = None
    
    def __post_init__(self):
        """Initialize VAD if not provided."""
        if self.vad is None:
            self.vad = VoiceActivityDetector()
    
    def load_response(self, audio_path: str) -> None:
        """Load response audio from file."""
        logger.info(f"Loading response audio from {audio_path}")
        self.response_chunks = load_response_audio(audio_path)
        logger.info(f"Loaded {len(self.response_chunks)} audio chunks")
    
    def reset(self) -> None:
        """Reset conversation state for a new call."""
        self.state = ConversationState.LISTENING
        self.vad.reset()
        self._chunk_index = 0
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
        self._playback_task = None
    
    async def process_audio(self, audio_bytes: bytes) -> None:
        """
        Process incoming audio from Twilio.
        
        This is called for each audio chunk received via WebSocket.
        The audio is decoded, resampled, and fed to the VAD.
        
        Args:
            audio_bytes: Raw mulaw audio bytes from Twilio (base64 decoded)
        """
        from .audio import decode_mulaw
        
        # Decode mulaw to PCM
        pcm = decode_mulaw(audio_bytes)
        
        # Upsample from 8kHz to 16kHz for VAD
        pcm_16k = upsample_for_vad(pcm)
        
        # Get speech state from VAD
        speech_state = self.vad.process(pcm_16k)
        
        # Handle state transitions
        await self._handle_speech_state(speech_state)
    
    async def _handle_speech_state(self, speech_state: SpeechState) -> None:
        """Handle speech state changes from VAD."""
        
        if self.state == ConversationState.LISTENING:
            # In listening state, wait for user to finish speaking
            if speech_state == SpeechState.SPEECH_END:
                logger.info("User finished speaking, starting playback")
                await self._start_playback()
            elif speech_state == SpeechState.SPEECH_START:
                logger.info("User started speaking")
        
        elif self.state == ConversationState.PLAYING:
            # In playing state, check for interrupts
            if speech_state == SpeechState.SPEECH_START:
                logger.info("User interrupted, stopping playback")
                await self._stop_playback()
    
    async def _start_playback(self) -> None:
        """Start playing the response audio."""
        if not self.response_chunks:
            logger.warning("No response audio loaded")
            return
        
        self.state = ConversationState.PLAYING
        self._chunk_index = 0
        
        # Start playback in background task
        self._playback_task = asyncio.create_task(self._playback_loop())
    
    async def _stop_playback(self) -> None:
        """Stop playback immediately (interrupt)."""
        # Cancel playback task
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
        
        self._playback_task = None
        
        # Send clear message to Twilio to stop buffered audio
        if self.send_clear:
            await self.send_clear()
        
        # Return to listening state
        self.state = ConversationState.LISTENING
        self._chunk_index = 0
        logger.info("Playback stopped, returning to listening")
    
    async def _playback_loop(self) -> None:
        """Background task that streams audio chunks to Twilio."""
        try:
            while self._chunk_index < len(self.response_chunks):
                if self.send_audio:
                    chunk = self.response_chunks[self._chunk_index]
                    await self.send_audio(chunk)
                
                self._chunk_index += 1
                
                # ~20ms per chunk at 8kHz (160 samples)
                # Slight delay to match real-time playback
                await asyncio.sleep(0.018)
            
            # Playback complete
            logger.info("Playback complete")
            self.state = ConversationState.LISTENING
            self._chunk_index = 0
            
        except asyncio.CancelledError:
            # Playback was interrupted
            raise
