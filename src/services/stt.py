"""
Deepgram Speech-to-Text service using the official SDK.

Uses AsyncDeepgramClient with the v1 listen WebSocket API.
"""

import os
import asyncio
from typing import Optional, Callable, Awaitable

from deepgram import AsyncDeepgramClient

from ..log import ServiceLogger

log = ServiceLogger("STT")


class STTService:
    """
    Deepgram streaming STT service.
    
    Audio should be mulaw at 8kHz (Twilio format).
    """
    
    def __init__(
        self,
        on_partial: Callable[[str], Awaitable[None]],
        on_final: Callable[[str], Awaitable[None]],
    ):
        self._on_partial = on_partial
        self._on_final = on_final
        
        self._client: Optional[AsyncDeepgramClient] = None
        self._connection = None
        self._listener_task: Optional[asyncio.Task] = None
        self._running = False
        
        self._transcript_parts: list = []
        self._audio_bytes_sent = 0
        
        self._api_key = os.getenv("DEEPGRAM_API_KEY", "")
    
    @property
    def is_active(self) -> bool:
        return self._running and self._connection is not None
    
    async def start(self) -> None:
        """Open connection to Deepgram."""
        if self._running:
            return
        
        self._transcript_parts = []
        self._audio_bytes_sent = 0
        
        try:
            self._client = AsyncDeepgramClient(api_key=self._api_key)
            
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
            
            self._connection.on("message", self._on_message)
            self._connection.on("UtteranceEnd", self._on_utterance_end_event)
            self._connection.on("SpeechStarted", self._on_speech_started_event)
            self._connection.on("Error", self._on_error_event)
            
            self._listener_task = asyncio.create_task(self._connection.start_listening())
            
            self._running = True
            log.connected()
            
        except Exception as e:
            log.error("Connection failed", e)
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
            log.error("Send failed", e)
    
    async def stop(self) -> None:
        """Close connection gracefully and send final transcript."""
        if not self._running:
            return
        
        log.debug(f"Sent {self._audio_bytes_sent} bytes total")
        
        # Wait for Deepgram to return final results
        wait_time = 0
        max_wait = 2.0
        while wait_time < max_wait and not self._transcript_parts:
            await asyncio.sleep(0.1)
            wait_time += 0.1
        
        self._running = False
        
        # Send accumulated transcript as final
        final_transcript = " ".join(self._transcript_parts).strip()
        if final_transcript:
            await self._on_final(final_transcript)
        
        await self._cleanup()
        log.disconnected()
    
    async def cancel(self) -> None:
        """Abort connection immediately without waiting for final."""
        self._running = False
        self._transcript_parts = []
        await self._cleanup()
        log.cancelled()
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
        
        if self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._cm = None
        
        self._connection = None
        self._client = None
    
    async def _on_message(self, result, *args, **kwargs):
        """Handle transcript result from Deepgram."""
        try:
            channel = getattr(result, 'channel', None)
            if not channel:
                return
            
            alternatives = getattr(channel, 'alternatives', None)
            if alternatives is None:
                alternatives = channel if isinstance(channel, list) else None
            
            if not alternatives:
                return
            
            alt = alternatives[0] if isinstance(alternatives, list) else alternatives
            transcript = getattr(alt, 'transcript', None) or (alt.get('transcript') if isinstance(alt, dict) else None)
            
            if not transcript:
                return
            
            is_final = getattr(result, 'is_final', False)
            
            if is_final:
                self._transcript_parts.append(transcript)
            
            await self._on_partial(transcript)
            
        except Exception as e:
            log.error("Message handling failed", e)
    
    async def _on_utterance_end_event(self, *args, **kwargs):
        """Handle utterance end event."""
        pass
    
    async def _on_speech_started_event(self, *args, **kwargs):
        """Handle speech started event."""
        pass
    
    async def _on_error_event(self, error, *args, **kwargs):
        """Handle error from Deepgram."""
        log.error(f"Deepgram error: {error}")
