#!/usr/bin/env python3
"""
shuo - Voice Agent Framework

Usage:
    python main.py +1234567890

This will:
1. Start the FastAPI server on the configured port
2. Initiate an outbound call to the specified phone number
3. Handle the call with VAD, STT, LLM, and TTS
"""

import os
import sys
import threading
import time

import uvicorn
from dotenv import load_dotenv

from shuo.server import app
from shuo.services.twilio_client import make_outbound_call
from shuo.log import setup_logging, Lifecycle, get_logger

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
    # Check for phone number argument
    if len(sys.argv) < 2:
        print("Usage: python main.py +1234567890")
        print("  Phone number must be in E.164 format")
        sys.exit(1)
    
    phone_number = sys.argv[1]
    
    # Validate phone number format
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
    Lifecycle.server_starting(port)
    server_thread = threading.Thread(
        target=start_server,
        args=(port,),
        daemon=True
    )
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    Lifecycle.server_ready(public_url)
    
    # Make outbound call
    Lifecycle.call_initiating(phone_number)
    try:
        call_sid = make_outbound_call(phone_number)
        Lifecycle.call_initiated(call_sid)
        logger.info("Waiting for call to connect... (Ctrl+C to end)")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        Lifecycle.shutdown()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
