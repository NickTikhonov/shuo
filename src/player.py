"""
Audio player for streaming audio to Twilio.

Manages its own independent playback loop that drips audio
chunks at the correct rate, regardless of other activity.
"""

import json
import asyncio
import logging
from typing import List, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class AudioPlayer:
    """
    Streams audio to Twilio at the correct rate.
    
    Features:
    - Independent playback loop (not affected by incoming messages)
    - Can be topped up with audio chunks
    - Instant stop and clear on interrupt
    """
    
    def __init__(self, websocket: WebSocket, stream_sid: str):
        self._websocket = websocket
        self._stream_sid = stream_sid
        self._chunks: List[str] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._index = 0
    
    @property
    def is_playing(self) -> bool:
        """Whether playback is currently active."""
        return self._running and self._task is not None and not self._task.done()
    
    async def play(self, chunks: List[str]) -> None:
        """
        Start playing audio chunks.
        
        Args:
            chunks: List of base64-encoded mulaw audio chunks
        """
        # Stop any existing playback
        if self.is_playing:
            await self.stop_and_clear()
        
        self._chunks = list(chunks)
        self._index = 0
        self._running = True
        
        logger.info(f"Starting playback of {len(self._chunks)} chunks")
        self._task = asyncio.create_task(self._playback_loop())
    
    async def stop_and_clear(self) -> None:
        """
        Stop playback immediately and clear Twilio's buffer.
        
        Called when user interrupts.
        """
        self._running = False
        
        # Cancel the playback task
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        self._task = None
        self._chunks = []
        self._index = 0
        
        # Clear Twilio's audio buffer for instant silence
        await self._send_clear()
        logger.info("Playback stopped and cleared")
    
    async def wait_until_done(self) -> None:
        """Wait for playback to complete (or be interrupted)."""
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _playback_loop(self) -> None:
        """
        Independent loop that drips audio at ~20ms intervals.
        
        This runs as a separate task, unaffected by other activity.
        """
        try:
            while self._index < len(self._chunks) and self._running:
                chunk = self._chunks[self._index]
                await self._send_audio(chunk)
                self._index += 1
                
                # 20ms per chunk at 8kHz (160 samples)
                await asyncio.sleep(0.020)
            
            if self._running:
                logger.info("Playback complete")
                self._running = False
                
        except asyncio.CancelledError:
            # Playback was interrupted
            raise
        except Exception as e:
            logger.error(f"Playback error: {e}")
            self._running = False
    
    async def _send_audio(self, payload: str) -> None:
        """Send a single audio chunk to Twilio."""
        message = {
            "event": "media",
            "streamSid": self._stream_sid,
            "media": {
                "payload": payload
            }
        }
        await self._websocket.send_text(json.dumps(message))
    
    async def _send_clear(self) -> None:
        """Send clear message to Twilio to flush audio buffer."""
        message = {
            "event": "clear",
            "streamSid": self._stream_sid
        }
        await self._websocket.send_text(json.dumps(message))
