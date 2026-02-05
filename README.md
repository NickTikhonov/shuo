# shuo 说

A real-time voice agent framework built from scratch. Makes phone calls, listens, thinks, and speaks.

> **shuo** (说) — Mandarin for "to speak"

## What It Does

Call someone, have a conversation:

```bash
python main.py +1234567890
```

The system:
1. **Listens** — Detects when you start/stop speaking (Silero VAD)
2. **Transcribes** — Converts speech to text in real-time (Deepgram)
3. **Thinks** — Generates a response (OpenAI GPT-4o-mini)
4. **Speaks** — Synthesizes audio and plays it back (ElevenLabs)
5. **Interrupts** — If you speak while it's talking, it stops immediately

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
│           await execute(action)           # Side effects            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    Events from multiple sources
                                   │
        ┌──────────────┬──────────┴──────────┬──────────────┐
        ▼              ▼                     ▼              ▼
   ┌─────────┐   ┌──────────┐         ┌──────────┐   ┌─────────┐
   │  Twilio │   │ Deepgram │         │  OpenAI  │   │ Eleven  │
   │WebSocket│   │   STT    │         │   LLM    │   │Labs TTS │
   └─────────┘   └──────────┘         └──────────┘   └─────────┘
```

### The Key Insight

**All business logic lives in a single pure function:**

```python
def update(state: AppState, event: Event) -> Tuple[AppState, List[Action]]:
    ...
```

- **No I/O** — Just state in, state + actions out
- **Testable** — 50 unit tests run in 1 second
- **Debuggable** — Print state at any point
- **Predictable** — Same input = same output

## State Machine

```
                    ┌─────────────────────┐
     ┌─────────────►│     LISTENING       │◄────────────────┐
     │              │   (VAD watching)    │                 │
     │              └──────────┬──────────┘                 │
     │                         │                            │
     │              user stops │ speaking                   │
     │              STT final  │ transcript                 │
     │                         ▼                            │
     │              ┌─────────────────────┐                 │
     │              │    PROCESSING       │                 │
     │   interrupt  │  (LLM + TTS streaming)               │  playback
     │   (barge-in) └──────────┬──────────┘                 │  complete
     │                         │                            │
     │              first TTS  │ audio arrives              │
     │                         ▼                            │
     │              ┌─────────────────────┐                 │
     └──────────────┤     SPEAKING        ├─────────────────┘
                    │  (playing audio)    │
                    └─────────────────────┘
```

### Interrupt Handling (Barge-in)

When the user speaks during PROCESSING or SPEAKING:
1. Cancel LLM generation
2. Cancel TTS synthesis
3. Clear Twilio's audio buffer (instant silence)
4. Start fresh STT session
5. Transition to LISTENING

This happens in ~50ms.

## Project Structure

```
shuo/
├── main.py                 # Entry point - starts server, makes call
├── requirements.txt        # Dependencies
├── tests/
│   ├── test_update.py      # 33 tests for state machine
│   └── test_vad.py         # 17 tests for VAD logic
└── src/
    ├── types.py            # State, Events, Actions (immutable dataclasses)
    ├── update.py           # Pure state machine - THE BRAIN
    ├── loop.py             # Main event loop - THE HEART
    ├── effects.py          # Side effect executor - THE HANDS
    ├── player.py           # Audio playback manager
    ├── vad.py              # Voice Activity Detection (Silero)
    ├── audio.py            # Codec utilities (mulaw ↔ PCM)
    ├── server.py           # FastAPI endpoints
    ├── twilio_client.py    # Outbound call initiation
    └── services/
        ├── stt.py          # Deepgram streaming STT
        ├── llm.py          # OpenAI streaming LLM
        └── tts.py          # ElevenLabs streaming TTS
```

## How It Works

### 1. Audio Flow

```
Twilio (mulaw 8kHz) 
    → decode to PCM 
    → upsample to 16kHz 
    → Silero VAD 
    → speech probability
```

### 2. Turn Detection

VAD uses **hysteresis** to prevent flickering:
- `START_PATIENCE = 250ms` — Must speak for 250ms before we "believe" it
- `END_PATIENCE = 700ms` — Must be silent for 700ms before turn ends

### 3. Pipeline Streaming

For minimum latency, everything streams in parallel:

```
User speaks → STT transcribes (streaming)
                    ↓
           STT final transcript
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
- `STTFinalEvent` — Transcription complete
- `LLMTokenEvent` — Token from GPT
- `TTSAudioEvent` — Audio chunk from ElevenLabs
- `PlaybackDoneEvent` — Finished playing

**Actions** (outputs/side effects):
- `StartSTTAction`, `FeedSTTAction`, `StopSTTAction`
- `StartLLMAction`, `CancelLLMAction`
- `StartTTSAction`, `FeedTTSAction`, `FlushTTSAction`
- `StartPlaybackAction`, `StopPlaybackAction`

## Setup

### Prerequisites

- Python 3.9+
- [ngrok](https://ngrok.com/) for exposing local server
- API keys for: Twilio, Deepgram, OpenAI, ElevenLabs

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

# 50 tests in ~1 second
```

Tests cover:
- State transitions (LISTENING → PROCESSING → SPEAKING)
- Interrupt handling (barge-in cancels everything)
- Conversation history management
- VAD hysteresis logic
- Edge cases and error handling

## Design Principles

1. **Functional Core, Imperative Shell**  
   All logic in pure `update()`, all I/O in `execute()`

2. **Immutable State**  
   All dataclasses are `frozen=True` — state is never mutated

3. **Single Event Queue**  
   All sources (Twilio, STT, LLM, TTS) push to one queue

4. **Explicit Everything**  
   No hidden callbacks, no implicit state, no magic

## License

MIT
