#!/usr/bin/env python3
"""
Voice Agent - Main Entry Point

Usage:
    python main.py +1234567890

This will:
1. Start the FastAPI server on the configured port
2. Initiate an outbound call to the specified phone number
3. Handle the call with:
   - VAD-based turn-taking
   - STT transcription (Deepgram)
   - LLM response generation (OpenAI)
   - TTS synthesis (ElevenLabs)
"""

import os
import sys
import logging
import threading
import time

import uvicorn
from dotenv import load_dotenv

from src.server import app
from src.twilio_client import make_outbound_call

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
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
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Please check your .env file")
        return False
    
    return True


def start_server(port: int) -> None:
    """Start the FastAPI server in a background thread."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()


def main():
    """
    Starts the server (main loop)
    Creates an outbound call to the specified phone number.
    """
    # Check for phone number argument
    if len(sys.argv) < 2:
        print("Usage: python main.py +1234567890")
        print("  Phone number must be in E.164 format (e.g., +1234567890)")
        sys.exit(1)
    
    phone_number = sys.argv[1]
    
    # Validate phone number format
    if not phone_number.startswith("+"):
        print("Error: Phone number must be in E.164 format (start with +)")
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Get port from environment
    port = int(os.getenv("PORT", "3040"))
    
    # Start server in background thread
    logger.info(f"Starting server on port {port}")
    server_thread = threading.Thread(
        target=start_server,
        args=(port,),
        daemon=True
    )
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Make outbound call
    logger.info(f"Calling {phone_number}...")
    try:
        call_sid = make_outbound_call(phone_number)
        logger.info(f"Call initiated successfully - SID: {call_sid}")
        logger.info("Waiting for call to connect...")
        logger.info("Press Ctrl+C to end")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
