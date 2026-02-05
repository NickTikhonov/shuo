"""
Twilio client for initiating outbound calls.

Uses the Twilio REST API to create calls that connect to our
TwiML endpoint, which then starts the Media Stream.
"""

import os
import logging

from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_twilio_client() -> Client:
    """Create and return a Twilio client."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    
    if not account_sid or not auth_token:
        raise ValueError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set")
    
    return Client(account_sid, auth_token)


def make_outbound_call(to_number: str) -> str:
    """
    Initiate an outbound call to the specified number.
    
    Args:
        to_number: Phone number to call (E.164 format, e.g., +1234567890)
        
    Returns:
        Call SID of the initiated call
    """
    client = get_twilio_client()
    
    from_number = os.getenv("TWILIO_PHONE_NUMBER")
    public_url = os.getenv("TWILIO_PUBLIC_URL")
    
    if not from_number:
        raise ValueError("TWILIO_PHONE_NUMBER must be set")
    if not public_url:
        raise ValueError("TWILIO_PUBLIC_URL must be set")
    
    twiml_url = f"{public_url}/twiml"
    
    logger.info(f"Initiating call from {from_number} to {to_number}")
    logger.info(f"TwiML URL: {twiml_url}")
    
    call = client.calls.create(
        to=to_number,
        from_=from_number,
        url=twiml_url,
    )
    
    logger.info(f"Call initiated - SID: {call.sid}")
    return call.sid
