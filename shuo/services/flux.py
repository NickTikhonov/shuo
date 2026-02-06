"""
Deepgram Flux service -- always-on STT + turn detection.

A single persistent WebSocket to Deepgram using the v2 listen API.
Receives all Twilio audio continuously and emits turn events.

Replaces both local VAD (Silero) and separate STT (Deepgram v1).
"""

import os
import asyncio
from typing import Optional, Callable, Awaitable

from deepgram import AsyncDeepgramClient

from ..log import ServiceLogger

log = ServiceLogger("Flux")


class FluxService:
    """
    Deepgram Flux streaming service.

    Audio format: mulaw 8kHz (direct from Twilio, no conversion needed).
    Turn events: StartOfTurn (barge-in), EndOfTurn (with transcript).
    """

    def __init__(
        self,
        on_end_of_turn: Callable[[str], Awaitable[None]],
        on_start_of_turn: Callable[[], Awaitable[None]],
        on_interim: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self._on_end_of_turn = on_end_of_turn
        self._on_start_of_turn = on_start_of_turn
        self._on_interim = on_interim

        self._api_key = os.getenv("DEEPGRAM_API_KEY", "")
        self._client: Optional[AsyncDeepgramClient] = None
        self._connection = None
        self._cm = None
        self._listener_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def is_active(self) -> bool:
        return self._running and self._connection is not None

    async def start(self) -> None:
        """Connect to Deepgram Flux (always-on for the duration of the call)."""
        if self._running:
            return

        try:
            self._client = AsyncDeepgramClient(api_key=self._api_key)

            self._cm = self._client.listen.v2.connect(
                model="flux-general-en",
                encoding="mulaw",
                sample_rate=8000,
            )
            self._connection = await self._cm.__aenter__()

            self._connection.on("message", self._on_message)
            self._connection.on("Error", self._on_error)

            self._listener_task = asyncio.create_task(
                self._connection.start_listening()
            )

            self._running = True
            log.connected()

        except Exception as e:
            log.error("Connection failed", e)
            await self._cleanup()
            raise

    async def send(self, audio_bytes: bytes) -> None:
        """Send audio chunk to Deepgram Flux."""
        if not self._connection or not self._running:
            return

        try:
            await self._connection.send_media(audio_bytes)
        except Exception as e:
            log.error("Send failed", e)

    async def stop(self) -> None:
        """Disconnect from Deepgram Flux."""
        self._running = False
        await self._cleanup()
        log.disconnected()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._running = False

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

    async def _on_message(self, message, *args, **kwargs) -> None:
        """Handle Flux messages -- parse TurnInfo events."""
        try:
            msg_type = getattr(message, "type", None)

            if msg_type == "TurnInfo":
                event = getattr(message, "event", None)

                if event == "EndOfTurn":
                    transcript = getattr(message, "transcript", "") or ""
                    await self._on_end_of_turn(transcript.strip())

                elif event == "StartOfTurn":
                    await self._on_start_of_turn()

            elif msg_type == "Results" and self._on_interim:
                channel = getattr(message, "channel", None)
                if channel:
                    alternatives = getattr(channel, "alternatives", None)
                    if alternatives:
                        alt = (
                            alternatives[0]
                            if isinstance(alternatives, list)
                            else alternatives
                        )
                        transcript = getattr(alt, "transcript", "")
                        if transcript:
                            await self._on_interim(transcript.strip())

        except Exception as e:
            log.error("Message handling failed", e)

    async def _on_error(self, error, *args, **kwargs) -> None:
        """Handle Deepgram errors."""
        log.error("Deepgram: " + str(error))
