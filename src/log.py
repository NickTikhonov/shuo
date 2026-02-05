"""
Centralized logging for shuo.

Provides:
- Configured console logger with colors
- EventLogger for consistent event logging
- Lifecycle logging helpers
"""

import logging
import sys
from typing import Optional

from .types import (
    Event,
    StreamStartEvent, StreamStopEvent, MediaEvent,
    STTPartialEvent, STTFinalEvent,
    LLMTokenEvent, LLMDoneEvent,
    TTSAudioEvent, TTSDoneEvent,
    PlaybackDoneEvent,
    Action,
    StartSTTAction, FeedSTTAction, StopSTTAction, CancelSTTAction,
    StartLLMAction, CancelLLMAction,
    StartTTSAction, FeedTTSAction, FlushTTSAction, CancelTTSAction,
    StartPlaybackAction, StopPlaybackAction,
    Phase,
)


# =============================================================================
# COLORS
# =============================================================================

class C:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


def _c(color: str, text: str) -> str:
    """Wrap text in color codes."""
    return color + text + C.RESET


def _quote(text: str, color: str = C.WHITE) -> str:
    """Wrap text in quotes with color."""
    return _c(color, '"' + text + '"')


# =============================================================================
# LOGGING SETUP
# =============================================================================

class ColorFormatter(logging.Formatter):
    """Custom formatter with colors and clean timestamp."""
    
    def format(self, record: logging.LogRecord) -> str:
        time_str = _c(C.DIM, self.formatTime(record, "%H:%M:%S"))
        return time_str + " │ " + record.getMessage()


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application."""
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColorFormatter())
    console.setLevel(level)
    
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [console]
    
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# =============================================================================
# LIFECYCLE LOGGING
# =============================================================================

class Lifecycle:
    """Lifecycle event logging with colors."""
    
    _logger = logging.getLogger("shuo")
    
    @classmethod
    def server_starting(cls, port: int) -> None:
        msg = _c(C.CYAN, "Server starting on port " + str(port))
        cls._logger.info("🚀 " + msg)
    
    @classmethod
    def server_ready(cls, url: str) -> None:
        cls._logger.info(_c(C.GREEN, "✓  Ready") + " " + _c(C.DIM, url))
    
    @classmethod
    def call_initiating(cls, phone: str) -> None:
        cls._logger.info("📞 " + _c(C.CYAN, "Calling " + phone + "..."))
    
    @classmethod
    def call_initiated(cls, sid: str) -> None:
        cls._logger.info(_c(C.GREEN, "✓  Call initiated") + " " + _c(C.DIM, "SID: " + sid[:8] + "..."))
    
    @classmethod
    def websocket_connected(cls) -> None:
        cls._logger.info("🔌 " + _c(C.CYAN, "WebSocket connected"))
    
    @classmethod
    def websocket_disconnected(cls) -> None:
        cls._logger.info("🔌 " + _c(C.DIM, "WebSocket disconnected"))
    
    @classmethod
    def stream_started(cls, stream_sid: str) -> None:
        cls._logger.info(_c(C.GREEN, "▶  Stream started") + " " + _c(C.DIM, "SID: " + stream_sid[:8] + "..."))
    
    @classmethod
    def stream_stopped(cls) -> None:
        cls._logger.info("⏹  " + _c(C.DIM, "Stream stopped"))
    
    @classmethod
    def shutdown(cls) -> None:
        cls._logger.info("👋 " + _c(C.DIM, "Shutting down"))


# =============================================================================
# EVENT LOGGING
# =============================================================================

class EventLogger:
    """Logs events in a consistent, readable format with colors."""
    
    def __init__(self, verbose: bool = False):
        self._logger = logging.getLogger("shuo.events")
        self._verbose = verbose
        self._llm_buffer = ""
    
    def event(self, event: Event) -> None:
        """Log an incoming event (green arrow)."""
        
        if isinstance(event, MediaEvent):
            if self._verbose:
                size = len(event.audio_bytes)
                self._logger.debug(_c(C.DIM, "← MediaEvent (" + str(size) + " bytes)"))
            return
        
        if isinstance(event, (StreamStartEvent, StreamStopEvent)):
            return  # Handled by Lifecycle
        
        if isinstance(event, STTPartialEvent):
            self._logger.debug(_c(C.DIM, "← STT partial: ") + _quote(event.text, C.DIM))
            return
        
        if isinstance(event, STTFinalEvent):
            self._logger.info(_c(C.GREEN, "←") + " " + _c(C.GREEN, "STT") + " " + _quote(event.text))
            return
        
        if isinstance(event, LLMTokenEvent):
            self._llm_buffer += event.token
            return
        
        if isinstance(event, LLMDoneEvent):
            response = self._llm_buffer.strip()
            if len(response) > 60:
                response = response[:57] + "..."
            self._logger.info(_c(C.GREEN, "←") + " " + _c(C.GREEN, "LLM") + " " + _quote(response))
            self._llm_buffer = ""
            return
        
        if isinstance(event, TTSAudioEvent):
            if self._verbose:
                size = len(event.audio_base64)
                self._logger.debug(_c(C.DIM, "← TTS audio (" + str(size) + " chars)"))
            return
        
        if isinstance(event, TTSDoneEvent):
            self._logger.info(_c(C.GREEN, "←") + " " + _c(C.DIM, "TTS done"))
            return
        
        if isinstance(event, PlaybackDoneEvent):
            self._logger.info(_c(C.GREEN, "←") + " " + _c(C.DIM, "Playback done"))
            return
    
    def action(self, action: Action) -> None:
        """Log an outgoing action (yellow arrow)."""
        
        if isinstance(action, FeedSTTAction):
            if self._verbose:
                size = len(action.audio_bytes)
                self._logger.debug(_c(C.DIM, "→ FeedSTT (" + str(size) + " bytes)"))
            return
        
        if isinstance(action, FeedTTSAction):
            return  # Don't log individual tokens
        
        if isinstance(action, StartSTTAction):
            self._logger.info(_c(C.YELLOW, "→") + " " + _c(C.YELLOW, "Start") + " " + _c(C.BRIGHT_BLUE, "STT"))
            return
        
        if isinstance(action, StopSTTAction):
            self._logger.info(_c(C.YELLOW, "→") + " " + _c(C.DIM, "Stop") + " " + _c(C.BRIGHT_BLUE, "STT"))
            return
        
        if isinstance(action, CancelSTTAction):
            self._logger.info(_c(C.YELLOW, "→") + " " + _c(C.DIM, "Cancel") + " " + _c(C.BRIGHT_BLUE, "STT"))
            return
        
        if isinstance(action, StartLLMAction):
            msg = action.user_message
            if len(msg) > 40:
                msg = msg[:37] + "..."
            self._logger.info(
                _c(C.YELLOW, "→") + " " + _c(C.YELLOW, "Start") + " " + 
                _c(C.BRIGHT_MAGENTA, "LLM") + " " + _quote(msg, C.DIM)
            )
            return
        
        if isinstance(action, CancelLLMAction):
            self._logger.info(_c(C.YELLOW, "→") + " " + _c(C.DIM, "Cancel") + " " + _c(C.BRIGHT_MAGENTA, "LLM"))
            return
        
        if isinstance(action, StartTTSAction):
            self._logger.info(_c(C.YELLOW, "→") + " " + _c(C.YELLOW, "Start") + " " + _c(C.BRIGHT_CYAN, "TTS"))
            return
        
        if isinstance(action, FlushTTSAction):
            self._logger.info(_c(C.YELLOW, "→") + " " + _c(C.DIM, "Flush") + " " + _c(C.BRIGHT_CYAN, "TTS"))
            return
        
        if isinstance(action, CancelTTSAction):
            self._logger.info(_c(C.YELLOW, "→") + " " + _c(C.DIM, "Cancel") + " " + _c(C.BRIGHT_CYAN, "TTS"))
            return
        
        if isinstance(action, StartPlaybackAction):
            self._logger.info(_c(C.YELLOW, "→") + " " + _c(C.YELLOW, "Start") + " " + _c(C.WHITE, "Playback"))
            return
        
        if isinstance(action, StopPlaybackAction):
            self._logger.info(_c(C.YELLOW, "→") + " " + _c(C.DIM, "Stop") + " " + _c(C.WHITE, "Playback"))
            return
    
    def transition(self, old_phase: Phase, new_phase: Phase) -> None:
        """Log a phase transition (magenta)."""
        if old_phase != new_phase:
            self._logger.info(
                _c(C.MAGENTA, "◆") + " " +
                _c(C.DIM, old_phase.name) + " " +
                _c(C.MAGENTA, "→") + " " +
                _c(C.BRIGHT_MAGENTA, new_phase.name)
            )
    
    def interrupt(self) -> None:
        """Log an interrupt (red)."""
        self._logger.info(_c(C.BRIGHT_RED, "⚡ INTERRUPT") + " " + _c(C.DIM, "(user spoke)"))
    
    def error(self, msg: str, exc: Optional[Exception] = None) -> None:
        """Log an error (red)."""
        if exc:
            self._logger.error(_c(C.RED, "✗ " + msg + ":") + " " + _c(C.DIM, str(exc)))
        else:
            self._logger.error(_c(C.RED, "✗ " + msg))


# =============================================================================
# SERVICE LOGGING
# =============================================================================

class ServiceLogger:
    """Logger for individual services (STT, LLM, TTS)."""
    
    COLORS = {
        "STT": C.BRIGHT_BLUE,
        "LLM": C.BRIGHT_MAGENTA,
        "TTS": C.BRIGHT_CYAN,
        "Player": C.WHITE,
    }
    
    def __init__(self, service_name: str):
        self._logger = logging.getLogger("shuo." + service_name)
        self._name = service_name
        self._color = self.COLORS.get(service_name, C.WHITE)
    
    def connected(self) -> None:
        self._logger.info(_c(C.GREEN, "✓") + " " + _c(self._color, self._name) + " " + _c(C.DIM, "connected"))
    
    def disconnected(self) -> None:
        self._logger.debug(_c(C.DIM, "○ " + self._name + " disconnected"))
    
    def cancelled(self) -> None:
        self._logger.debug(_c(C.DIM, "○ " + self._name + " cancelled"))
    
    def error(self, msg: str, exc: Optional[Exception] = None) -> None:
        if exc:
            self._logger.error(_c(C.RED, "✗") + " " + _c(self._color, self._name + ":") + " " + msg + " " + _c(C.DIM, "(" + str(exc) + ")"))
        else:
            self._logger.error(_c(C.RED, "✗") + " " + _c(self._color, self._name + ":") + " " + msg)
    
    def debug(self, msg: str) -> None:
        self._logger.debug("  " + _c(C.DIM, self._name + ": " + msg))
    
    def info(self, msg: str) -> None:
        self._logger.info("  " + _c(self._color, self._name + ":") + " " + msg)
