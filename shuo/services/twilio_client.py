"""
Twilio service -- outbound calls and WebSocket message parsing.
"""

import os
import json
import base64
from typing import Optional

from twilio.rest import Client

from ..types import (
    Event, StreamStartEvent, StreamStopEvent, MediaEvent,
)
from ..log import Logger


def make_outbound_call(to_number: str) -> str:
    """
    Initiate an outbound call using Twilio.
    
    Args:
        to_number: Phone number to call in E.164 format (+1234567890)
        
    Returns:
        Call SID
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_PHONE_NUMBER")
    public_url = os.getenv("TWILIO_PUBLIC_URL")
    
    if not all([account_sid, auth_token, from_number, public_url]):
        raise ValueError("Missing required Twilio environment variables")
    
    client = Client(account_sid, auth_token)
    
    twiml_url = f"{public_url}/twiml"
    
    call = client.calls.create(
        to=to_number,
        from_=from_number,
        url=twiml_url,
    )
    
    return call.sid


def parse_twilio_message(data: dict) -> Optional[Event]:
    """Parse raw Twilio WebSocket message into typed Event."""
    event_type = data.get("event")

    if event_type == "connected":
        Logger.websocket_connected()
        return None

    elif event_type == "start":
        start_data = data.get("start", {})
        stream_sid = start_data.get("streamSid")
        if stream_sid:
            return StreamStartEvent(stream_sid=stream_sid)

    elif event_type == "media":
        media_data = data.get("media", {})
        payload = media_data.get("payload", "")
        if payload:
            audio_bytes = base64.b64decode(payload)
            return MediaEvent(audio_bytes=audio_bytes)

    elif event_type == "stop":
        return StreamStopEvent()

    return None
