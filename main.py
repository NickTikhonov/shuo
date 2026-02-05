#!/usr/bin/env python3
"""
Voice Agent VAD System - Main Entry Point

Usage:
    python main.py +1234567890

This will:
1. Start the FastAPI server on the configured port
2. Initiate an outbound call to the specified phone number
3. Handle the call with VAD-based turn-taking
"""

import os
import sys
import asyncio
import logging
import threading
import time
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from src.server import app, init_conversation_manager
from src.twilio_client import make_outbound_call

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    """Main entry point."""
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
    
    # Get port from environment
    port = int(os.getenv("PORT", "3040"))
    
    # Check for response audio
    response_audio_path = Path(__file__).parent / "static" / "response.wav"
    if not response_audio_path.exists():
        print(f"Error: Response audio not found at {response_audio_path}")
        print("Please create a response.wav file in the static/ directory")
        sys.exit(1)
    
    # Initialize conversation manager with response audio
    logger.info(f"Loading response audio from {response_audio_path}")
    init_conversation_manager(str(response_audio_path))
    
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
