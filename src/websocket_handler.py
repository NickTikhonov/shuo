"""
Twilio Media Stream WebSocket handler.

Handles the bidirectional audio stream between Twilio and our VAD system.
Processes incoming audio, sends outgoing audio, and manages stream lifecycle.
"""

import json
import base64
import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import WebSocket

from .conversation import ConversationManager

logger = logging.getLogger(__name__)


@dataclass
class TwilioMediaStreamHandler:
    """
    Handles Twilio Media Stream WebSocket connections.
    
    Twilio Media Streams send JSON messages with these event types:
    - connected: WebSocket connection established
    - start: Media stream starting, contains stream metadata
    - media: Audio data (base64 mulaw)
    - stop: Media stream ending
    """
    
    websocket: WebSocket
    conversation: ConversationManager
    
    # Stream metadata from Twilio
    stream_sid: Optional[str] = None
    call_sid: Optional[str] = None
    
    async def handle(self) -> None:
        """Main handler loop for WebSocket messages."""
        
        # Set up callbacks for the conversation manager
        self.conversation.send_audio = self._send_audio
        self.conversation.send_clear = self._send_clear
        
        try:
            while True:
                message = await self.websocket.receive_text()
                await self._process_message(message)
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            raise
    
    async def _process_message(self, message: str) -> None:
        """Process a single WebSocket message from Twilio."""
        data = json.loads(message)
        event = data.get("event")
        
        if event == "connected":
            await self._handle_connected(data)
        elif event == "start":
            await self._handle_start(data)
        elif event == "media":
            await self._handle_media(data)
        elif event == "stop":
            await self._handle_stop(data)
        else:
            logger.debug(f"Unknown event: {event}")
    
    async def _handle_connected(self, data: dict) -> None:
        """Handle connection established event."""
        logger.info("Twilio Media Stream connected")
    
    async def _handle_start(self, data: dict) -> None:
        """Handle stream start event with metadata."""
        start_data = data.get("start", {})
        
        self.stream_sid = start_data.get("streamSid")
        self.call_sid = start_data.get("callSid")
        
        # Store stream SID in conversation manager
        self.conversation.stream_sid = self.stream_sid
        
        logger.info(f"Stream started - StreamSID: {self.stream_sid}, CallSID: {self.call_sid}")
        
        # Reset conversation state for new call
        self.conversation.reset()
    
    async def _handle_media(self, data: dict) -> None:
        """Handle incoming audio data."""
        media_data = data.get("media", {})
        payload = media_data.get("payload", "")
        
        if not payload:
            return
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(payload)
        
        # Process through conversation manager
        await self.conversation.process_audio(audio_bytes)
    
    async def _handle_stop(self, data: dict) -> None:
        """Handle stream stop event."""
        logger.info("Twilio Media Stream stopped")
    
    async def _send_audio(self, audio_b64: str) -> None:
        """
        Send audio chunk to Twilio.
        
        Args:
            audio_b64: Base64-encoded mulaw audio
        """
        if not self.stream_sid:
            logger.warning("Cannot send audio: no stream SID")
            return
        
        message = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {
                "payload": audio_b64
            }
        }
        
        await self.websocket.send_text(json.dumps(message))
    
    async def _send_clear(self) -> None:
        """
        Send clear message to Twilio to stop buffered audio.
        
        This immediately stops any audio that's queued for playback,
        enabling instant interruption when the user starts speaking.
        """
        if not self.stream_sid:
            logger.warning("Cannot send clear: no stream SID")
            return
        
        message = {
            "event": "clear",
            "streamSid": self.stream_sid
        }
        
        await self.websocket.send_text(json.dumps(message))
        logger.debug("Sent clear message to Twilio")
