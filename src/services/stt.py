"""
Deepgram Speech-to-Text service using the official SDK.

Based on working implementation using AsyncDeepgramClient.
Uses the context manager pattern with event handlers.
"""

import os
import asyncio
import logging
from typing import Optional, Callable, Awaitable

from deepgram import AsyncDeepgramClient

logger = logging.getLogger(__name__)


class STTService:
    """
    Deepgram streaming STT service using official SDK.
    
    Uses AsyncDeepgramClient with the v1 listen WebSocket API.
    Audio should be mulaw at 8kHz (Twilio format).
    """
    
    def __init__(
        self,
        on_partial: Callable[[str], Awaitable[None]],
        on_final: Callable[[str], Awaitable[None]],
    ):
        """
        Args:
            on_partial: Callback for interim transcriptions
            on_final: Callback for final transcriptions (utterance complete)
        """
        self._on_partial = on_partial
        self._on_final = on_final
        
        self._client: Optional[AsyncDeepgramClient] = None
        self._connection = None
        self._listener_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Accumulate transcript segments for final
        self._transcript_parts: list = []
        self._audio_bytes_sent = 0
        
        # Get API key
        self._api_key = os.getenv("DEEPGRAM_API_KEY", "")
        if not self._api_key:
            logger.warning("DEEPGRAM_API_KEY not set")
    
    @property
    def is_active(self) -> bool:
        """Whether STT is currently streaming."""
        return self._running and self._connection is not None
    
    async def start(self) -> None:
        """Open connection to Deepgram."""
        if self._running:
            logger.warning("STT already running")
            return
        
        # Reset state
        self._transcript_parts = []
        self._audio_bytes_sent = 0
        
        try:
            # Create client
            self._client = AsyncDeepgramClient(api_key=self._api_key)
            
            # Open connection using context manager
            # We manually enter it here to keep the connection open
            self._cm = self._client.listen.v1.connect(
                model="nova-2",
                language="en-US",
                encoding="mulaw",
                sample_rate="8000",
                channels="1",
                smart_format="true",
                interim_results="true",
                utterance_end_ms="1000",
                vad_events="true",
            )
            self._connection = await self._cm.__aenter__()
            
            # Register event handlers
            self._connection.on("message", self._on_message)
            self._connection.on("UtteranceEnd", self._on_utterance_end_event)
            self._connection.on("SpeechStarted", self._on_speech_started_event)
            self._connection.on("Error", self._on_error_event)
            
            # Start listening for responses
            self._listener_task = asyncio.create_task(self._connection.start_listening())
            
            self._running = True
            logger.info("STT connection opened")
            
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            await self._cleanup()
            raise
    
    async def send(self, audio_bytes: bytes) -> None:
        """Send audio chunk to Deepgram."""
        if not self._connection or not self._running:
            return
        
        try:
            await self._connection.send_media(audio_bytes)
            self._audio_bytes_sent += len(audio_bytes)
        except Exception as e:
            logger.error(f"Error sending audio to Deepgram: {e}")
    
    async def stop(self) -> None:
        """
        Close connection gracefully and send final transcript.
        """
        if not self._running:
            return
        
        self._running = False
        logger.info(f"STT: Sent {self._audio_bytes_sent} bytes of audio total")
        
        # Send accumulated transcript as final
        final_transcript = " ".join(self._transcript_parts).strip()
        if final_transcript:
            logger.info(f"STT final transcript: {final_transcript}")
            await self._on_final(final_transcript)
        else:
            logger.warning("STT: No transcript accumulated")
        
        await self._cleanup()
        logger.info("STT connection closed")
    
    async def cancel(self) -> None:
        """Abort connection immediately without waiting for final."""
        self._running = False
        self._transcript_parts = []
        await self._cleanup()
        logger.info("STT connection cancelled")
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        # Cancel listener task
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
        
        # Exit context manager
        if self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._cm = None
        
        self._connection = None
        self._client = None
    
    # --- Event Handlers ---
    
    async def _on_message(self, result, *args, **kwargs):
        """Handle transcript result from Deepgram."""
        try:
            # The SDK passes result as an object with channel attribute
            # But the channel contains a list of alternatives, not an 'alternatives' attribute
            channel = getattr(result, 'channel', None)
            if not channel:
                return
            
            # Channel might have alternatives as an attribute or as a list
            alternatives = getattr(channel, 'alternatives', None)
            if alternatives is None:
                # Try treating channel as the alternatives list
                alternatives = channel if isinstance(channel, list) else None
            
            if not alternatives:
                return
            
            # Get first alternative
            alt = alternatives[0] if isinstance(alternatives, list) else alternatives
            transcript = getattr(alt, 'transcript', None) or (alt.get('transcript') if isinstance(alt, dict) else None)
            
            if not transcript:
                return
            
            is_final = getattr(result, 'is_final', False)
            speech_final = getattr(result, 'speech_final', False)
            
            logger.debug(f"STT {'final' if is_final else 'partial'}: {transcript}")
            
            if is_final:
                # Accumulate final segments
                self._transcript_parts.append(transcript)
            
            # Fire callback
            await self._on_partial(transcript)
            
        except Exception as e:
            logger.error(f"Error handling transcript: {e}", exc_info=True)
    
    async def _on_utterance_end_event(self, *args, **kwargs):
        """Handle utterance end event from Deepgram."""
        logger.debug("Deepgram: Utterance end detected")
    
    async def _on_speech_started_event(self, *args, **kwargs):
        """Handle speech started event from Deepgram."""
        logger.debug("Deepgram: Speech started")
    
    async def _on_error_event(self, error, *args, **kwargs):
        """Handle error from Deepgram."""
        logger.error(f"Deepgram error: {error}")
