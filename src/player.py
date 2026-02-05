"""
Audio player for streaming audio to Twilio.

Manages its own independent playback loop that drips audio
chunks at the correct rate, regardless of other activity.

For streaming TTS, chunks are added dynamically as they arrive.
"""

import json
import asyncio
import logging
from typing import List, Optional, Callable

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class AudioPlayer:
    """
    Streams audio to Twilio at the correct rate.
    
    Features:
    - Independent playback loop (not affected by incoming messages)
    - Can be topped up with audio chunks dynamically (for streaming TTS)
    - Instant stop and clear on interrupt
    - Callback when playback completes
    """
    
    def __init__(
        self,
        websocket: WebSocket,
        stream_sid: str,
        on_done: Optional[Callable[[], None]] = None,
    ):
        self._websocket = websocket
        self._stream_sid = stream_sid
        self._on_done = on_done
        
        self._chunks: List[str] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._index = 0
        self._tts_done = False  # Flag to know when TTS is complete
    
    @property
    def is_playing(self) -> bool:
        """Whether playback is currently active."""
        return self._running and self._task is not None and not self._task.done()
    
    async def start(self) -> None:
        """
        Start the playback loop.
        
        Chunks will be added via send_chunk() as TTS produces them.
        """
        if self.is_playing:
            await self.stop_and_clear()
        
        self._chunks = []
        self._index = 0
        self._running = True
        self._tts_done = False
        
        logger.info("Starting playback (streaming mode)")
        self._task = asyncio.create_task(self._playback_loop())
    
    async def send_chunk(self, chunk: str) -> None:
        """
        Add an audio chunk to the playback queue.
        
        Called by TTS service as audio is generated.
        
        Args:
            chunk: Base64-encoded mulaw audio chunk
        """
        if not self._running:
            # Start playback if not already running
            await self.start()
        
        self._chunks.append(chunk)
    
    def mark_tts_done(self) -> None:
        """
        Signal that TTS is complete - no more chunks coming.
        
        Playback will finish when all chunks are sent.
        """
        self._tts_done = True
    
    async def play(self, chunks: List[str]) -> None:
        """
        Start playing a fixed list of audio chunks.
        
        Used for pre-recorded audio (legacy mode).
        
        Args:
            chunks: List of base64-encoded mulaw audio chunks
        """
        if self.is_playing:
            await self.stop_and_clear()
        
        self._chunks = list(chunks)
        self._index = 0
        self._running = True
        self._tts_done = True  # All chunks provided upfront
        
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
        self._tts_done = False
        
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
        For streaming TTS, it waits for chunks to arrive.
        """
        try:
            while self._running:
                # Check if we have a chunk to send
                if self._index < len(self._chunks):
                    chunk = self._chunks[self._index]
                    await self._send_audio(chunk)
                    self._index += 1
                    
                    # 20ms per chunk at 8kHz (160 samples)
                    await asyncio.sleep(0.020)
                    
                elif self._tts_done:
                    # TTS is done and we've sent all chunks
                    break
                else:
                    # Waiting for more chunks from TTS
                    await asyncio.sleep(0.010)
            
            if self._running:
                logger.info("Playback complete")
                self._running = False
                
                # Notify that playback is done
                if self._on_done:
                    self._on_done()
                
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
