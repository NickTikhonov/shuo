"""
External services for the shuo voice agent pipeline.

Deepgram Flux  -- STT + turn detection
OpenAI         -- LLM streaming
ElevenLabs     -- TTS streaming + connection pool
Twilio         -- outbound calls + audio playback
"""

from .flux import FluxService
from .llm import LLMService
from .tts import TTSService
from .tts_pool import TTSPool
from .player import AudioPlayer
from .twilio_client import make_outbound_call

__all__ = [
    "FluxService",
    "LLMService",
    "TTSService",
    "TTSPool",
    "AudioPlayer",
    "make_outbound_call",
]
