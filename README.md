# Voice Agent VAD System

A Voice Activity Detection (VAD) system built on top of Twilio phone calls. This is the foundation for building a complete voice agent.

## Architecture

The system uses a **functional architecture** with an explicit event loop:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MAIN LOOP (loop.py)                           │
│                                                                         │
│   while connected:                                                      │
│       event = receive()                    # I/O - WebSocket/Timer     │
│       state, actions = update(state, event) # PURE - No side effects  │
│       for action in actions:                                           │
│           execute(action)                  # I/O - Send to Twilio     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Functional?

- **Testable**: `update()` is pure - just call it with state and event, assert the result
- **Debuggable**: Print state before/after any transition
- **Extensible**: Add STT/LLM/TTS by adding new Events and Actions
- **Predictable**: All state in one place, all side effects isolated

## State Machine

```
                         ┌─────────────────────┐
                         │                     │
    ┌───────────────────►│     LISTENING       │◄────────────────┐
    │                    │                     │                 │
    │                    └──────────┬──────────┘                 │
    │                               │                            │
    │                    speech_end │                            │
    │                               ▼                            │
    │                    ┌─────────────────────┐                 │
    │   interrupt        │                     │    playback     │
    │   (speech_start)   │      PLAYING        │    complete     │
    └────────────────────┤                     ├─────────────────┘
                         └─────────────────────┘
```

## Project Structure

```
/vapi_clone
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── main.py                # CLI entry point
├── static/
│   └── response.wav       # Pre-recorded response audio
└── src/
    ├── types.py           # State, Events, Actions (dataclasses)
    ├── vad.py             # Voice Activity Detection (pure functions)
    ├── audio.py           # Audio codec utilities (pure functions)
    ├── update.py          # State machine (pure function)
    ├── effects.py         # Side effects (WebSocket I/O)
    ├── loop.py            # Main event loop
    ├── server.py          # FastAPI endpoints
    └── twilio_client.py   # Outbound call initiation
```

## Key Files Explained

### `types.py` - The Data Model
All state is immutable dataclasses:
- `AppState`: Complete application state
- `Event`: Things that happen (MediaEvent, StreamStartEvent, etc.)
- `Action`: Side effects to perform (SendAudioAction, ClearBufferAction, etc.)

### `update.py` - The Brain
Pure function: `(State, Event) -> (State, List[Action])`
- No I/O, no side effects
- All business logic in one place
- Easy to test and reason about

### `loop.py` - The Heart
Explicit event loop:
1. RECEIVE event from WebSocket or timer
2. UPDATE state (call pure function)
3. EXECUTE actions (perform side effects)

### `effects.py` - The Boundary
The ONLY place with side effects:
- Send audio to Twilio
- Send clear message
- Manage playback timer

## Agent Rules

1. **Functional Core, Imperative Shell**: Pure logic in `update.py`, I/O in `effects.py`
2. **Immutable State**: All dataclasses are `frozen=True`
3. **Explicit Loop**: No hidden callbacks, everything flows through the main loop
4. **Clean Python**: Type hints, small functions, meaningful names

## Setup

### Prerequisites

- Python 3.9+
- Twilio account with a phone number
- ngrok for local development

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create `.env`:
```
PORT=3040
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890
TWILIO_PUBLIC_URL=https://your-subdomain.ngrok-free.app
```

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

## Extending to Full Voice Agent

The architecture is designed for easy extension:

```python
# Add new phases
class Phase(Enum):
    LISTENING = auto()
    TRANSCRIBING = auto()  # NEW: STT processing
    THINKING = auto()       # NEW: LLM generating
    SYNTHESIZING = auto()   # NEW: TTS generating
    PLAYING = auto()

# Add new events
@dataclass(frozen=True)
class TranscriptReadyEvent:
    text: str

@dataclass(frozen=True)
class LLMResponseEvent:
    text: str

# Add new actions
@dataclass(frozen=True)
class TranscribeAction:
    audio_chunks: Tuple[bytes, ...]

@dataclass(frozen=True)
class GenerateResponseAction:
    transcript: str
    history: Tuple[Turn, ...]
```

The `update()` function grows with new match cases, but stays pure.
The `execute()` function handles new action types with async service calls.
