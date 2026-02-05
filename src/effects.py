"""
Side effects executor for shuo.

This is the ONLY place where I/O happens.
Maps Actions to service method calls.
"""

from .types import (
    Action,
    StartSTTAction, FeedSTTAction, StopSTTAction, CancelSTTAction,
    StartLLMAction, CancelLLMAction,
    StartTTSAction, FeedTTSAction, FlushTTSAction, CancelTTSAction,
    StartPlaybackAction, StopPlaybackAction,
)
from .player import AudioPlayer
from .services.stt import STTService
from .services.llm import LLMService
from .services.tts import TTSService


class EffectsExecutor:
    """
    Executes side effects by coordinating with services.
    
    This is the boundary between pure logic and the outside world.
    """
    
    def __init__(
        self,
        player: AudioPlayer,
        stt: STTService,
        llm: LLMService,
        tts: TTSService,
    ):
        self._player = player
        self._stt = stt
        self._llm = llm
        self._tts = tts
    
    async def execute(self, action: Action) -> None:
        """Execute a single action."""
        
        # --- STT Actions ---
        if isinstance(action, StartSTTAction):
            await self._stt.start()
        
        elif isinstance(action, FeedSTTAction):
            await self._stt.send(action.audio_bytes)
        
        elif isinstance(action, StopSTTAction):
            await self._stt.stop()
        
        elif isinstance(action, CancelSTTAction):
            await self._stt.cancel()
        
        # --- LLM Actions ---
        elif isinstance(action, StartLLMAction):
            await self._llm.start(action.user_message)
        
        elif isinstance(action, CancelLLMAction):
            await self._llm.cancel()
        
        # --- TTS Actions ---
        elif isinstance(action, StartTTSAction):
            await self._tts.start()
        
        elif isinstance(action, FeedTTSAction):
            await self._tts.send(action.text)
        
        elif isinstance(action, FlushTTSAction):
            await self._tts.flush()
        
        elif isinstance(action, CancelTTSAction):
            await self._tts.cancel()
        
        # --- Playback Actions ---
        elif isinstance(action, StartPlaybackAction):
            pass  # Audio is fed directly from TTS to player
        
        elif isinstance(action, StopPlaybackAction):
            await self._player.stop_and_clear()
