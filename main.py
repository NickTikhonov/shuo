#!/usr/bin/env python3
"""
shuo - Voice Agent Framework

Usage:
    python main.py                  # server-only mode (inbound calls)
    python main.py +1234567890      # outbound call mode

Server-only mode starts the server and waits for inbound calls.
Outbound mode additionally initiates a call to the specified number.
"""

import os
import sys
import signal
import threading
import time

import uvicorn
from dotenv import load_dotenv

from shuo.server import app
from shuo.services.twilio_client import make_outbound_call
from shuo.log import setup_logging, Logger, get_logger
import shuo.server as server_module

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger("shuo")


def check_environment() -> bool:
    """Check that all required environment variables are set."""
    required_vars = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_PHONE_NUMBER",
        "TWILIO_PUBLIC_URL",
        "DEEPGRAM_API_KEY",
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        return False
    
    return True


# Max time (seconds) to wait for active calls to finish before forced exit.
DRAIN_TIMEOUT = int(os.getenv("DRAIN_TIMEOUT", "300"))  # 5 minutes default

_uvicorn_server: uvicorn.Server = None


def start_server(port: int) -> None:
    """Start the FastAPI server."""
    global _uvicorn_server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",  # Quiet uvicorn, we have our own logging
    )
    _uvicorn_server = uvicorn.Server(config)
    _uvicorn_server.run()


def main():
    """Main entry point."""
    phone_number = None

    if len(sys.argv) >= 2:
        phone_number = sys.argv[1]
        if not phone_number.startswith("+"):
            print("Error: Phone number must start with +")
            sys.exit(1)

    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Get port from environment
    port = int(os.getenv("PORT", "3040"))
    public_url = os.getenv("TWILIO_PUBLIC_URL", "")
    
    # Start server in background thread
    Logger.server_starting(port)
    server_thread = threading.Thread(
        target=start_server,
        args=(port,),
        daemon=True
    )
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    Logger.server_ready(public_url)
    
    # ── Graceful shutdown on SIGTERM ────────────────────────────────
    def _handle_sigterm(signum, frame):
        """
        Railway (and Docker) send SIGTERM before killing the container.
        We stop accepting new calls and wait for active ones to finish.
        """
        logger.info("SIGTERM received — starting graceful drain")
        server_module._draining = True

        # If no active calls, exit immediately
        if server_module._active_calls <= 0:
            logger.info("No active calls — shutting down now")
            if _uvicorn_server:
                _uvicorn_server.should_exit = True
            return

        logger.info(
            f"Waiting up to {DRAIN_TIMEOUT}s for {server_module._active_calls} "
            f"active call(s) to finish..."
        )

        # Poll until calls drain or timeout
        deadline = time.monotonic() + DRAIN_TIMEOUT
        while server_module._active_calls > 0 and time.monotonic() < deadline:
            time.sleep(1)

        remaining = server_module._active_calls
        if remaining > 0:
            logger.warning(f"Drain timeout — {remaining} call(s) still active, forcing exit")
        else:
            logger.info("All calls drained — shutting down cleanly")

        if _uvicorn_server:
            _uvicorn_server.should_exit = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        if phone_number:
            # Outbound call mode
            Logger.call_initiating(phone_number)
            call_sid = make_outbound_call(phone_number)
            Logger.call_initiated(call_sid)
            logger.info("Waiting for call to connect... (Ctrl+C to end)")
        else:
            # Server-only mode — wait for inbound calls
            logger.info("Server-only mode — waiting for inbound calls (Ctrl+C to end)")

        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        Logger.shutdown()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
