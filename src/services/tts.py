"""
ElevenLabs Text-to-Speech service.

WebSocket streaming API for real-time speech synthesis.
"""

import os
import json
import base64
import asyncio
import logging
from typing import Optional, Callable, Awaitable

import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


class TTSService:
    """
    ElevenLabs streaming TTS service.
    
    Manages a WebSocket connection for real-time synthesis.
    Sends text chunks, receives audio chunks via callback.
    
    Audio is returned as base64-encoded mulaw at 8kHz for Twilio.
    """
    
    def __init__(
        self,
        on_audio: Callable[[str], Awaitable[None]],  # base64 audio
        on_done: Callable[[], Awaitable[None]],
    ):
        """
        Args:
            on_audio: Callback for each audio chunk (base64 encoded)
            on_done: Callback when synthesis completes
        """
        self._on_audio = on_audio
        self._on_done = on_done
        
        self._ws: Optional[WebSocketClientProtocol] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._running = False
        
        self._api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self._voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default: Rachel
        
        if not self._api_key:
            logger.warning("ELEVENLABS_API_KEY not set")
    
    @property
    def is_active(self) -> bool:
        """Whether TTS is currently active."""
        return self._running and self._ws is not None
    
    async def start(self) -> None:
        """Open WebSocket connection to ElevenLabs."""
        if self._running:
            logger.warning("TTS already running")
            return
        
        # ElevenLabs WebSocket URL
        # - output_format=ulaw_8000 for Twilio compatibility
        url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream-input?"
            f"model_id=eleven_turbo_v2_5&"
            f"output_format=ulaw_8000"
        )
        
        try:
            self._ws = await websockets.connect(url)
            self._running = True
            
            # Send initial configuration
            init_message = {
                "text": " ",  # Initial space to start the stream
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                },
                "xi_api_key": self._api_key,
            }
            await self._ws.send(json.dumps(init_message))
            
            self._receive_task = asyncio.create_task(self._receive_loop())
            logger.info("TTS connection opened")
            
        except Exception as e:
            logger.error(f"Failed to connect to ElevenLabs: {e}")
            raise
    
    async def send(self, text: str) -> None:
        """Send text chunk for synthesis."""
        if not self._ws or not self._running:
            return
        
        try:
            message = {
                "text": text,
                "try_trigger_generation": True,
            }
            await self._ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending text to ElevenLabs: {e}")
    
    async def flush(self) -> None:
        """Force synthesis of any buffered text."""
        if not self._ws or not self._running:
            return
        
        try:
            # Send empty string with flush flag to trigger generation
            message = {
                "text": "",
                "flush": True,
            }
            await self._ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error flushing TTS: {e}")
    
    async def stop(self) -> None:
        """Close connection gracefully after flushing."""
        if not self._running:
            return
        
        try:
            await self.flush()
            # Give it a moment to process remaining audio
            await asyncio.sleep(0.2)
        except Exception as e:
            logger.error(f"Error stopping TTS: {e}")
        finally:
            await self._cleanup()
        
        logger.info("TTS connection closed")
    
    async def cancel(self) -> None:
        """Abort connection immediately."""
        self._running = False
        await self._cleanup()
        logger.info("TTS connection cancelled")
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._running = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
    
    async def _receive_loop(self) -> None:
        """Background task to receive audio chunks."""
        try:
            while self._running and self._ws:
                try:
                    message = await self._ws.recv()
                    await self._handle_message(message)
                except websockets.exceptions.ConnectionClosed:
                    logger.info("ElevenLabs connection closed")
                    break
                except Exception as e:
                    logger.error(f"Error receiving from ElevenLabs: {e}")
                    break
        finally:
            if self._running:
                self._running = False
                await self._on_done()
    
    async def _handle_message(self, message: str) -> None:
        """Parse and handle ElevenLabs response."""
        try:
            data = json.loads(message)
            
            # Check for audio data
            if "audio" in data and data["audio"]:
                # Audio is already base64 encoded
                audio_base64 = data["audio"]
                await self._on_audio(audio_base64)
            
            # Check for completion
            if data.get("isFinal", False):
                logger.info("TTS synthesis complete")
                await self._on_done()
            
            # Check for errors
            if "error" in data:
                logger.error(f"ElevenLabs error: {data['error']}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from ElevenLabs: {message}")
