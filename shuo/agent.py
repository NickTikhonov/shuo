"""
Agent -- self-contained LLM -> TTS -> Player pipeline.

Encapsulates the entire agent response lifecycle.
Owns conversation history across turns.

    start_turn(transcript) -> add to history -> LLM -> TTS -> Player -> Twilio
    cancel_turn()          -> cancel all, keep history

TTS connections are managed by TTSPool (see services/tts_pool.py).
"""

import asyncio
import time
from typing import Optional, Callable, List, Dict

from fastapi import WebSocket

from .services.llm import LLMService
from .services.tts import TTSService
from .services.tts_pool import TTSPool
from .services.player import AudioPlayer
from .tracer import Tracer
from .log import ServiceLogger

log = ServiceLogger("Agent")


def _ms_since(t0: float) -> int:
    """Milliseconds elapsed since t0."""
    return int((time.monotonic() - t0) * 1000)


class Agent:
    """
    Self-contained agent response pipeline.

    LLM is persistent (keeps conversation history across turns).
    TTS connections come from TTSPool (pre-connected, with TTL eviction).
    Player is created fresh per turn.
    """

    def __init__(
        self,
        websocket: WebSocket,
        stream_sid: str,
        on_done: Callable[[], None],
        tts_pool: TTSPool,
        tracer: Tracer,
    ):
        self._websocket = websocket
        self._stream_sid = stream_sid
        self._on_done = on_done
        self._tts_pool = tts_pool
        self._tracer = tracer

        # Persistent LLM -- keeps conversation history across turns
        self._llm = LLMService(
            on_token=self._on_llm_token,
            on_done=self._on_llm_done,
        )

        # Active per-turn services (set during start, cleared on cancel)
        self._tts: Optional[TTSService] = None
        self._player: Optional[AudioPlayer] = None
        self._active = False

        # Current turn number (for tracer)
        self._turn: int = 0

        # Latency milestones (monotonic timestamps, reset each turn)
        self._t0: float = 0.0
        self._t_tts_conn: float = 0.0
        self._t_first_token: float = 0.0
        self._t_first_audio: float = 0.0
        self._got_first_token = False
        self._got_first_audio = False

    @property
    def is_turn_active(self) -> bool:
        return self._active

    @property
    def history(self) -> List[Dict[str, str]]:
        """Read-only access to conversation history (owned by LLM)."""
        return self._llm.history

    # ── Turn Lifecycle ──────────────────────────────────────────────

    async def start_turn(self, transcript: str) -> None:
        """Start a new agent turn."""
        if self._active:
            await self.cancel_turn()

        self._active = True
        self._t0 = time.monotonic()
        self._got_first_token = False
        self._got_first_audio = False

        # Begin tracing this turn
        self._turn = self._tracer.begin_turn(transcript)
        self._tracer.begin(self._turn, "tts_pool")

        # Get TTS from pool (instant if warm, blocks if cold)
        self._tts = await self._tts_pool.get(
            on_audio=self._on_tts_audio,
            on_done=self._on_tts_done,
        )
        self._t_tts_conn = time.monotonic()
        self._tracer.end(self._turn, "tts_pool")

        # Create player
        self._player = AudioPlayer(
            websocket=self._websocket,
            stream_sid=self._stream_sid,
            on_done=self._on_playback_done,
        )

        # Start LLM
        self._tracer.begin(self._turn, "llm")
        await self._llm.start(transcript)

        tts_ms = int((self._t_tts_conn - self._t0) * 1000)
        log.info(f"Turn started  (TTS {tts_ms}ms = {tts_ms}ms setup)")

    async def cancel_turn(self) -> None:
        """Cancel current turn, preserve history."""
        if not self._active:
            return

        elapsed = _ms_since(self._t0) if self._t0 else 0
        self._active = False

        # Mark turn as cancelled (ends all open spans)
        self._tracer.cancel_turn(self._turn)

        # Cancel in order: LLM -> TTS -> Player
        await self._llm.cancel()

        if self._tts:
            await self._tts.cancel()
            self._tts = None

        if self._player:
            if self._player.is_playing:
                await self._player.stop_and_clear()
            self._player = None

        log.info(f"Turn cancelled at +{elapsed}ms (history preserved)")

    async def cleanup(self) -> None:
        """Final cleanup when call ends."""
        if self._active:
            await self.cancel_turn()

    # ── Internal Callbacks ──────────────────────────────────────────

    async def _on_llm_token(self, token: str) -> None:
        """LLM produced a token -> feed to TTS."""
        if not self._active or not self._tts:
            return

        if not self._got_first_token:
            self._got_first_token = True
            self._t_first_token = time.monotonic()
            self._tracer.mark(self._turn, "llm_first_token")
            self._tracer.begin(self._turn, "tts")
            log.info(f"⏱  LLM first token  +{_ms_since(self._t0)}ms")

        await self._tts.send(token)

    async def _on_llm_done(self) -> None:
        """LLM finished -> flush TTS."""
        if not self._active or not self._tts:
            return
        self._tracer.end(self._turn, "llm")
        await self._tts.flush()

    async def _on_tts_audio(self, audio_base64: str) -> None:
        """TTS produced audio -> send to player."""
        if not self._active or not self._player:
            return

        if not self._got_first_audio:
            self._got_first_audio = True
            self._t_first_audio = time.monotonic()
            self._tracer.mark(self._turn, "tts_first_audio")
            self._tracer.begin(self._turn, "player")
            ttft = _ms_since(self._t0)
            since_token = int((self._t_first_audio - self._t_first_token) * 1000) if self._got_first_token else 0
            log.info(f"⏱  TTS first audio  +{ttft}ms  (TTS latency {since_token}ms)")

        await self._player.send_chunk(audio_base64)

    async def _on_tts_done(self) -> None:
        """TTS finished -> tell player no more chunks coming."""
        if not self._active or not self._player:
            return
        self._tracer.end(self._turn, "tts")
        self._player.mark_tts_done()

    def _on_playback_done(self) -> None:
        """Player finished -> turn is complete."""
        if not self._active:
            return

        self._tracer.end(self._turn, "player")

        total = _ms_since(self._t0)
        log.info(f"⏱  Turn complete    +{total}ms total")

        self._active = False
        self._tts = None
        self._player = None

        self._on_done()
