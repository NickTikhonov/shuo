# Voice Agent VAD System

A Voice Activity Detection (VAD) system built on top of Twilio phone calls. This is the foundation for building a complete voice agent.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Phone Call                               │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Twilio Media Stream                   │    │
│  │                   (mulaw 8kHz base64)                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              WebSocket Handler (/ws)                     │    │
│  │         Decode audio, manage stream lifecycle            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Silero VAD (16kHz)                      │    │
│  │           Detect speech start/end events                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               Conversation State Machine                 │    │
│  │         LISTENING ←→ PLAYING (with interrupt)            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Rules

1. **Modular Design**: Code is organized into focused modules:
   - `audio.py` - Audio codec utilities
   - `vad.py` - Voice activity detection
   - `conversation.py` - State machine
   - `websocket_handler.py` - Twilio integration
   - `server.py` - FastAPI endpoints
   - `twilio_client.py` - Outbound calls

2. **Clean Python**: 
   - Type hints throughout
   - Dataclasses for configuration
   - Async/await for I/O operations
   - Comprehensive logging

3. **VAD State Machine**:
   - `LISTENING` → Wait for user to finish speaking
   - `PLAYING` → Stream response audio
   - Instant interrupt when user speaks during playback

4. **Configurable Thresholds**:
   - `threshold`: Speech probability cutoff (default: 0.5)
   - `start_patience`: Min speech duration to trigger (default: 250ms)
   - `end_patience`: Min silence duration to end turn (default: 700ms)

## Setup

### Prerequisites

- Python 3.11+
- Twilio account with a phone number
- ngrok for local development

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```
PORT=3040
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890
TWILIO_PUBLIC_URL=https://your-subdomain.ngrok-free.app
```

### ngrok Setup

Start ngrok to expose your local server:

```bash
ngrok http 3040
```

Update `TWILIO_PUBLIC_URL` in `.env` with the ngrok URL.

## Usage

```bash
python main.py +1234567890
```

This will:
1. Start the FastAPI server on port 3040
2. Call the specified phone number
3. When you answer, speak something and stop
4. The pre-recorded response will play
5. If you speak during playback, it stops immediately

## Project Structure

```
/vapi_clone
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── main.py                # CLI entry point
├── static/
│   └── response.wav       # Pre-recorded response audio
└── src/
    ├── __init__.py
    ├── audio.py           # mulaw encode/decode, resampling
    ├── vad.py             # Silero VAD wrapper
    ├── conversation.py    # State machine + audio playback
    ├── websocket_handler.py  # Twilio Media Stream handling
    ├── server.py          # FastAPI endpoints
    └── twilio_client.py   # Outbound call initiation
```

## How It Works

1. **Outbound Call**: `main.py` uses Twilio REST API to initiate a call
2. **TwiML**: When answered, Twilio fetches `/twiml` which starts a Media Stream
3. **WebSocket**: Twilio connects to `/ws` for bidirectional audio
4. **VAD Processing**: 
   - Audio arrives as base64 mulaw (8kHz)
   - Decoded and upsampled to 16kHz for Silero VAD
   - VAD outputs speech probability per 32ms window
5. **State Machine**:
   - Accumulates speech/silence duration
   - Triggers `SPEECH_END` after 700ms of silence
   - Starts audio playback
6. **Interrupt Handling**:
   - If speech detected during playback
   - Send `clear` message to Twilio (stops buffered audio)
   - Return to `LISTENING` state

## Next Steps

This VAD foundation can be extended with:
- Speech-to-text (Deepgram, Whisper)
- LLM processing (OpenAI, Claude)
- Text-to-speech (ElevenLabs, OpenAI)
- Dynamic response generation
