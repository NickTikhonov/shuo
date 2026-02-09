"""
ElevenLabs Text-to-Speech service with WebSocket streaming.
"""

import os
import json
import asyncio
from typing import Optional, Callable, Awaitable

import websockets
from websockets.client import WebSocketClientProtocol

from ..log import ServiceLogger

log = ServiceLogger("TTS")


class TTSService:
    """
    ElevenLabs streaming TTS service.
    
    Sends text chunks, receives audio chunks via callback.
    Audio is returned as base64-encoded mulaw at 8kHz for Twilio.
    """
    
    def __init__(
        self,
        on_audio: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
    ):
        self._on_audio = on_audio
        self._on_done = on_done
        
        self._ws: Optional[WebSocketClientProtocol] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._running = False
        
        self._api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self._voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    
    @property
    def is_active(self) -> bool:
        return self._running and self._ws is not None

    def bind(
        self,
        on_audio: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
    ) -> None:
        """Rebind callbacks (used by connection pool to assign per-turn handlers)."""
        self._on_audio = on_audio
        self._on_done = on_done
    
    async def start(self) -> None:
        """Open WebSocket connection to ElevenLabs."""
        if self._running:
            return
        
        url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream-input?"
            f"model_id=eleven_turbo_v2_5&"
            f"output_format=ulaw_8000"
        )
        
        try:
            self._ws = await websockets.connect(url)
            self._running = True

            # Log ElevenLabs region (expect "Netherlands" from DE)
            # websockets v15: response headers live on ws.response.headers
            resp = getattr(self._ws, "response", None)
            hdrs = getattr(resp, "headers", {}) if resp else {}
            region = hdrs.get("x-region", "unknown")
            log.info(f"Region: {region}")
            
            init_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                },
                "xi_api_key": self._api_key,
            }
            await self._ws.send(json.dumps(init_message))
            
            self._receive_task = asyncio.create_task(self._receive_loop())
            log.connected()
            
        except Exception as e:
            log.error("Connection failed", e)
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
            log.error("Send failed", e)
    
    async def flush(self) -> None:
        """Force synthesis of any buffered text."""
        if not self._ws or not self._running:
            return
        
        try:
            message = {
                "text": "",
                "flush": True,
            }
            await self._ws.send(json.dumps(message))
        except Exception as e:
            log.error("Flush failed", e)
    
    async def stop(self) -> None:
        """Close connection gracefully after flushing."""
        if not self._running:
            return
        
        try:
            await self.flush()
            await asyncio.sleep(0.2)
        except Exception as e:
            log.error("Stop failed", e)
        finally:
            await self._cleanup()
        
        log.disconnected()
    
    async def cancel(self) -> None:
        """Abort connection immediately."""
        self._running = False
        await self._cleanup()
        log.cancelled()
    
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
                    break
                except Exception as e:
                    log.error("Receive failed", e)
                    break
        finally:
            if self._running:
                self._running = False
                await self._on_done()
    
    async def _handle_message(self, message: str) -> None:
        """Parse and handle ElevenLabs response."""
        try:
            data = json.loads(message)
            
            if "audio" in data and data["audio"]:
                audio_base64 = data["audio"]
                await self._on_audio(audio_base64)
            
            if data.get("isFinal", False):
                await self._on_done()
            
        except json.JSONDecodeError:
            log.error(f"Invalid JSON: {message[:100]}")
