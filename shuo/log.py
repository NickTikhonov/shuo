"""
Centralized logging for shuo.

Provides:
- Configured console logger with colors
- Logger for consistent event/lifecycle/action logging
- ServiceLogger for individual services
"""

import logging
import sys
from typing import Optional

from .types import (
    Event,
    StreamStartEvent, StreamStopEvent, MediaEvent,
    FluxStartOfTurnEvent, FluxEndOfTurnEvent,
    AgentTurnDoneEvent,
    Action,
    FeedFluxAction, StartAgentTurnAction, ResetAgentTurnAction,
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
        # Millisecond-precision timestamps for latency debugging
        ms = int(record.msecs)
        ts = self.formatTime(record, "%H:%M:%S") + f".{ms:03d}"
        time_str = _c(C.DIM, ts)
        return time_str + " \u2502 " + record.getMessage()


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
# LOGGER (unified lifecycle + event + action logging)
# =============================================================================

class Logger:
    """
    Unified logger for shuo.

    Class methods  -- lifecycle events (server, call, websocket, stream)
    Instance methods -- event/action/transition logging in the conversation loop
    """

    _logger = logging.getLogger("shuo")

    # ── Lifecycle (class methods) ────────────────────────────────────

    @classmethod
    def server_starting(cls, port: int) -> None:
        cls._logger.info("\U0001F680 " + _c(C.CYAN, "Server starting on port " + str(port)))

    @classmethod
    def server_ready(cls, url: str) -> None:
        cls._logger.info(_c(C.GREEN, "\u2713  Ready") + " " + _c(C.DIM, url))

    @classmethod
    def call_initiating(cls, phone: str) -> None:
        cls._logger.info("\U0001F4DE " + _c(C.CYAN, "Calling " + phone + "..."))

    @classmethod
    def call_initiated(cls, sid: str) -> None:
        cls._logger.info(
            _c(C.GREEN, "\u2713  Call initiated") + " " + _c(C.DIM, "SID: " + sid[:8] + "...")
        )

    @classmethod
    def websocket_connected(cls) -> None:
        cls._logger.info("\U0001F50C " + _c(C.CYAN, "WebSocket connected"))

    @classmethod
    def websocket_disconnected(cls) -> None:
        cls._logger.info("\U0001F50C " + _c(C.DIM, "WebSocket disconnected"))

    @classmethod
    def shutdown(cls) -> None:
        cls._logger.info("\U0001F44B " + _c(C.DIM, "Shutting down"))

    # ── Instance methods (conversation loop) ─────────────────────────

    def __init__(self, verbose: bool = False):
        self._events_logger = logging.getLogger("shuo.events")
        self._verbose = verbose

    def event(self, event: Event) -> None:
        """Log an incoming event."""

        if isinstance(event, MediaEvent):
            if self._verbose:
                size = len(event.audio_bytes)
                self._events_logger.debug(_c(C.DIM, "\u2190 MediaEvent (" + str(size) + " bytes)"))
            return

        if isinstance(event, StreamStartEvent):
            self._events_logger.info(
                _c(C.GREEN, "\u25B6  Stream started") + " " +
                _c(C.DIM, "SID: " + event.stream_sid[:8] + "...")
            )
            return

        if isinstance(event, StreamStopEvent):
            self._events_logger.info("\u23F9  " + _c(C.DIM, "Stream stopped"))
            return

        if isinstance(event, FluxEndOfTurnEvent):
            text = event.transcript
            if len(text) > 60:
                text = text[:57] + "..."
            self._events_logger.info(
                _c(C.GREEN, "\u2190") + " " +
                _c(C.BRIGHT_BLUE, "Flux") + " " +
                _c(C.GREEN, "EndOfTurn") + " " +
                _quote(text)
            )
            return

        if isinstance(event, FluxStartOfTurnEvent):
            self._events_logger.info(
                _c(C.BRIGHT_RED, "\u26A1") + " " +
                _c(C.BRIGHT_BLUE, "Flux") + " " +
                _c(C.BRIGHT_RED, "StartOfTurn") + " " +
                _c(C.DIM, "(barge-in)")
            )
            return

        if isinstance(event, AgentTurnDoneEvent):
            self._events_logger.info(
                _c(C.GREEN, "\u2190") + " " +
                _c(C.DIM, "Agent turn done")
            )
            return

    def action(self, action: Action) -> None:
        """Log an outgoing action."""

        if isinstance(action, FeedFluxAction):
            if self._verbose:
                size = len(action.audio_bytes)
                self._events_logger.debug(_c(C.DIM, "\u2192 FeedFlux (" + str(size) + " bytes)"))
            return

        if isinstance(action, StartAgentTurnAction):
            msg = action.transcript
            if len(msg) > 40:
                msg = msg[:37] + "..."
            self._events_logger.info(
                _c(C.YELLOW, "\u2192") + " " +
                _c(C.YELLOW, "Start") + " " +
                _c(C.BRIGHT_CYAN, "Agent") + " " +
                _quote(msg, C.DIM)
            )
            return

        if isinstance(action, ResetAgentTurnAction):
            self._events_logger.info(
                _c(C.YELLOW, "\u2192") + " " +
                _c(C.BRIGHT_RED, "Reset") + " " +
                _c(C.BRIGHT_CYAN, "Agent")
            )
            return

    def transition(self, old_phase: Phase, new_phase: Phase) -> None:
        """Log a phase transition (magenta)."""
        if old_phase != new_phase:
            self._events_logger.info(
                _c(C.MAGENTA, "\u25C6") + " " +
                _c(C.DIM, old_phase.name) + " " +
                _c(C.MAGENTA, "\u2192") + " " +
                _c(C.BRIGHT_MAGENTA, new_phase.name)
            )

    def error(self, msg: str, exc: Optional[Exception] = None) -> None:
        """Log an error (red)."""
        if exc:
            self._events_logger.error(
                _c(C.RED, "\u2717 " + msg + ":") + " " + _c(C.DIM, str(exc))
            )
        else:
            self._events_logger.error(_c(C.RED, "\u2717 " + msg))


# =============================================================================
# SERVICE LOGGING
# =============================================================================

class ServiceLogger:
    """Logger for individual services (Flux, LLM, TTS, Player, Agent)."""

    COLORS = {
        "Flux": C.BRIGHT_BLUE,
        "LLM": C.BRIGHT_MAGENTA,
        "TTS": C.BRIGHT_CYAN,
        "Player": C.WHITE,
        "Agent": C.BRIGHT_GREEN,
    }

    def __init__(self, service_name: str):
        self._logger = logging.getLogger("shuo." + service_name)
        self._name = service_name
        self._color = self.COLORS.get(service_name, C.WHITE)

    def connected(self) -> None:
        self._logger.info(
            _c(C.GREEN, "\u2713") + " " + _c(self._color, self._name) + " " + _c(C.DIM, "connected")
        )

    def disconnected(self) -> None:
        self._logger.debug(_c(C.DIM, "\u25CB " + self._name + " disconnected"))

    def cancelled(self) -> None:
        self._logger.debug(_c(C.DIM, "\u25CB " + self._name + " cancelled"))

    def error(self, msg: str, exc: Optional[Exception] = None) -> None:
        if exc:
            self._logger.error(
                _c(C.RED, "\u2717") + " " +
                _c(self._color, self._name + ":") + " " +
                msg + " " + _c(C.DIM, "(" + str(exc) + ")")
            )
        else:
            self._logger.error(
                _c(C.RED, "\u2717") + " " + _c(self._color, self._name + ":") + " " + msg
            )

    def debug(self, msg: str) -> None:
        self._logger.debug("  " + _c(C.DIM, self._name + ": " + msg))

    def info(self, msg: str) -> None:
        self._logger.info("  " + _c(self._color, self._name + ":") + " " + msg)
