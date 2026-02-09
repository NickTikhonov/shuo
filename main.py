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
import threading
import time

import uvicorn
from dotenv import load_dotenv

from shuo.server import app
from shuo.services.twilio_client import make_outbound_call
from shuo.log import setup_logging, Logger, get_logger

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


def start_server(port: int) -> None:
    """Start the FastAPI server."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",  # Quiet uvicorn, we have our own logging
    )
    server = uvicorn.Server(config)
    server.run()


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
