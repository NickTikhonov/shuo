# shuo 说

A real-time voice agent framework built from scratch. Makes phone calls, listens, thinks, and speaks.

> **shuo** (说) — Mandarin for "to speak"

## What It Does

Call someone, have a conversation:

```bash
python main.py +1234567890
```

The system:
1. **Listens** — Streams audio to Deepgram Flux, which handles both speech recognition and turn detection
2. **Detects turns** — Flux emits `StartOfTurn` (user began speaking) and `EndOfTurn` (user finished, with transcript)
3. **Thinks** — Generates a response via OpenAI GPT-4o-mini (streaming)
4. **Speaks** — Synthesizes audio via ElevenLabs (streaming) and plays it back
5. **Interrupts** — If you speak while it's talking, Flux fires `StartOfTurn` and the agent stops immediately

All with sub-second latency through aggressive streaming and pipelining.

## Architecture

Shuo uses a **functional, event-driven architecture** inspired by Elm/Redux:

```
┌─────────────────────────────────────────────────────────────────────┐
│                           TWILIO PHONE                              │
│                      (User speaks into phone)                       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    Audio (mulaw 8kHz) via WebSocket
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MAIN EVENT LOOP                              │
│                                                                     │
│   while connected:                                                  │
│       event = await queue.get()           # From any source         │
│       state, actions = update(state, event)  # PURE FUNCTION       │
│       for action in actions:                                        │
│           dispatch(action)                # Side effects            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
               │                                     │
     Events from Flux                     Actions dispatched to
               │                                     │
        ┌──────┴──────┐              ┌──────────────┴──────────────┐
        ▼             ▼              ▼                             ▼
   ┌─────────┐  ┌──────────┐  ┌──────────┐                 ┌──────────┐
   │  Flux   │  │ AgentTurn│  │   Flux   │                 │AgentTurn │
   │StartOf  │  │  Done    │  │ FeedAudio│                 │Start/    │
   │Turn     │  │          │  │          │                 │Reset     │
   └─────────┘  └──────────┘  └──────────┘                 └──────────┘
```

### Two Key Abstractions

**1. Deepgram Flux** (`src/flux.py`) — Always-on STT + turn detection. Replaces local VAD and separate STT. A single persistent WebSocket receives all audio and emits turn events.

**2. AgentTurn** (`src/agent_turn.py`) — Self-contained response pipeline. Encapsulates LLM → TTS → Player as a single cancellable unit. Owns conversation history.

```
AgentTurn.start(transcript)  → add to history → LLM → TTS → Player → Twilio
AgentTurn.reset()            → cancel all, keep history
```

### The Key Insight

**All business logic lives in a single pure function (~30 lines):**

```python
def update(state: AppState, event: Event) -> Tuple[AppState, List[Action]]:
    ...
```

- **No I/O** — Just state in, state + actions out
- **Testable** — 24 unit tests run in 0.03 seconds
- **Debuggable** — Print state at any point
- **Predictable** — Same input = same output

## State Machine

```
              ┌─────────────────────────┐
   ┌──────────►       LISTENING         │◄──────────────┐
   │          │  (Flux receiving audio) │                │
   │          └───────────┬─────────────┘                │
   │                      │                              │
   │           Flux       │ EndOfTurn                    │ AgentTurn
   │           StartOf    │ (with transcript)            │ Done
   │           Turn       │                              │ (playback
   │           (barge-in) │                              │  complete)
   │                      ▼                              │
   │          ┌─────────────────────────┐                │
   └──────────┤       RESPONDING        ├────────────────┘
              │  (LLM → TTS → Player)  │
              └─────────────────────────┘
```

Only two phases. Only three actions. The entire state machine is trivial because Flux handles all the complexity of turn detection.

### Interrupt Handling (Barge-in)

When Flux detects `StartOfTurn` during RESPONDING:
1. Cancel LLM generation
2. Cancel TTS synthesis
3. Clear Twilio's audio buffer (instant silence)
4. Transition to LISTENING

History is preserved — the next turn has full conversation context.

## Project Structure

```
shuo/
├── main.py                 # Entry point — starts server, makes call
├── requirements.txt        # Dependencies
├── tests/
│   └── test_update.py      # 24 tests for state machine
└── src/
    ├── types.py            # State, Events, Actions (immutable dataclasses)
    ├── update.py           # Pure state machine — THE BRAIN (~30 lines)
    ├── loop.py             # Main event loop — THE HEART
    ├── flux.py             # Deepgram Flux — always-on STT + turns
    ├── agent_turn.py       # Response pipeline — LLM → TTS → Player
    ├── player.py           # Audio playback manager
    ├── audio.py            # Codec utilities (mulaw ↔ PCM)
    ├── server.py           # FastAPI endpoints
    ├── twilio_client.py    # Outbound call initiation
    ├── log.py              # Colored logging
    └── services/
        ├── llm.py          # OpenAI streaming LLM
        └── tts.py          # ElevenLabs streaming TTS
```

## How It Works

### 1. Audio Flow

```
Twilio (mulaw 8kHz) → Flux (always-on WebSocket) → Turn events
```

No local audio processing needed. Flux handles everything server-side.

### 2. Turn Detection

Flux uses a conversational speech recognition model built for voice agents:
- `StartOfTurn` — User began speaking (triggers barge-in if agent is talking)
- `EndOfTurn` — User finished speaking (includes full transcript)

No local VAD, no hysteresis tuning, no resampling.

### 3. Pipeline Streaming

For minimum latency, everything streams in parallel:

```
Flux EndOfTurn (transcript)
        ↓
  LLM generates (streaming tokens)
        ↓ (each token)
  TTS synthesizes (streaming audio)
        ↓ (each chunk)
  Player sends to Twilio
```

LLM tokens are forwarded to TTS immediately — we don't wait for the full response.

### 4. Events & Actions

**Events** (inputs to the system):
- `MediaEvent` — Audio from Twilio
- `FluxEndOfTurnEvent` — User finished speaking (with transcript)
- `FluxStartOfTurnEvent` — User started speaking (barge-in)
- `AgentTurnDoneEvent` — Agent finished playing response

**Actions** (outputs/side effects):
- `FeedFluxAction` — Send audio to Deepgram Flux
- `StartAgentTurnAction` — Start the LLM → TTS → Player pipeline
- `ResetAgentTurnAction` — Cancel everything and clear Twilio buffer

## Setup

### Prerequisites

- Python 3.9+
- [ngrok](https://ngrok.com/) for exposing local server
- API keys for: Twilio, Deepgram (with Flux access), OpenAI, ElevenLabs

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create `.env`:

```bash
PORT=3040

# Twilio
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+1234567890
TWILIO_PUBLIC_URL=https://your-subdomain.ngrok-free.app

# AI Services
DEEPGRAM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ELEVENLABS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
```

### Running

```bash
# Start ngrok (in another terminal)
ngrok http 3040

# Update TWILIO_PUBLIC_URL in .env with ngrok URL

# Make a call
python main.py +1234567890
```

## Testing

The pure functional core is fully testable without mocking:

```bash
# Run all tests
python -m pytest tests/ -v

# 24 tests in ~0.03 seconds
```

Tests cover:
- State transitions (LISTENING ↔ RESPONDING)
- Interrupt handling (barge-in resets agent)
- Audio routing (always forwarded to Flux)
- Complete conversation flows
- Edge cases and state immutability

## Design Principles

1. **Functional Core, Imperative Shell**
   All logic in pure `update()`, all I/O in services

2. **Two Abstractions**
   Flux (input) and AgentTurn (output) — that's the whole system

3. **Immutable State**
   All dataclasses are `frozen=True` — state is never mutated

4. **Single Event Queue**
   All sources (Twilio, Flux, AgentTurn) push to one queue

5. **Conversation History in AgentTurn**
   History survives interrupts — `reset()` cancels pipeline but keeps context

## License

MIT
