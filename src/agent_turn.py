"""
Agent turn -- self-contained LLM -> TTS -> Player pipeline.

Encapsulates the entire agent response lifecycle.
Owns conversation history across turns.

    start(transcript) -> add to history -> LLM -> TTS -> Player -> Twilio
    reset()           -> cancel all, keep history
"""

import asyncio
from typing import Optional, Callable, List, Dict

from fastapi import WebSocket

from .services.llm import LLMService
from .services.tts import TTSService
from .player import AudioPlayer
from .log import ServiceLogger

log = ServiceLogger("AgentTurn")


class AgentTurn:
    """
    Self-contained agent response pipeline.

    LLM is persistent (keeps conversation history across turns).
    TTS and Player are created fresh per turn.
    """

    def __init__(
        self,
        websocket: WebSocket,
        stream_sid: str,
        on_done: Callable[[], None],
    ):
        self._websocket = websocket
        self._stream_sid = stream_sid
        self._on_done = on_done

        # Persistent LLM -- keeps conversation history across turns
        self._llm = LLMService(
            on_token=self._on_llm_token,
            on_done=self._on_llm_done,
        )

        # Per-turn services (created in start, destroyed in reset)
        self._tts: Optional[TTSService] = None
        self._player: Optional[AudioPlayer] = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def history(self) -> List[Dict[str, str]]:
        """Read-only access to conversation history (owned by LLM)."""
        return self._llm.history

    async def start(self, transcript: str) -> None:
        """Start a new agent turn."""
        if self._active:
            await self.reset()

        self._active = True

        # Create per-turn services
        self._tts = TTSService(
            on_audio=self._on_tts_audio,
            on_done=self._on_tts_done,
        )

        self._player = AudioPlayer(
            websocket=self._websocket,
            stream_sid=self._stream_sid,
            on_done=self._on_playback_done,
        )

        # Start pipeline: TTS first (opens WebSocket), then LLM
        await self._tts.start()
        await self._llm.start(transcript)

        log.info("Turn started")

    async def reset(self) -> None:
        """Cancel current turn, preserve history."""
        if not self._active:
            return

        self._active = False

        # Cancel in order: LLM -> TTS -> Player
        await self._llm.cancel()

        if self._tts:
            await self._tts.cancel()
            self._tts = None

        if self._player:
            if self._player.is_playing:
                await self._player.stop_and_clear()
            self._player = None

        log.info("Turn reset (history preserved)")

    async def cleanup(self) -> None:
        """Final cleanup when call ends."""
        if self._active:
            await self.reset()

    # ── Internal Callbacks ──────────────────────────────────────────

    async def _on_llm_token(self, token: str) -> None:
        """LLM produced a token -> feed to TTS."""
        if not self._active or not self._tts:
            return
        await self._tts.send(token)

    async def _on_llm_done(self) -> None:
        """LLM finished -> flush TTS."""
        if not self._active or not self._tts:
            return
        await self._tts.flush()

    async def _on_tts_audio(self, audio_base64: str) -> None:
        """TTS produced audio -> send to player."""
        if not self._active or not self._player:
            return
        await self._player.send_chunk(audio_base64)

    async def _on_tts_done(self) -> None:
        """TTS finished -> tell player no more chunks coming."""
        if not self._active or not self._player:
            return
        self._player.mark_tts_done()

    def _on_playback_done(self) -> None:
        """Player finished -> turn is complete."""
        if not self._active:
            return

        self._active = False
        self._tts = None
        self._player = None

        self._on_done()
