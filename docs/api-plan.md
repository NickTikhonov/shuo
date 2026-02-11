# Secure Configurable Call API

## Context

The project currently has a bare `GET /call/{phone_number}` endpoint with no auth and hardcoded config (system prompt, LLM model, voice, etc). We need a proper API with:
- **API key auth** (keys from env vars, Bearer token)
- **Per-call config** passed inline (system prompt, LLM model/provider, voice, first message, max duration, recording)
- **No database** -- all in-memory, config per-request

The core challenge: when `POST /v1/calls` creates a Twilio call, Twilio calls back to `/twiml` → `/ws` seconds later. The WebSocket handler needs to know which config to use. Solution: in-memory `call_sid → config` registry.

## API Design

```
POST   /v1/calls           -- Launch a call (auth required)
GET    /v1/calls/{call_id}  -- Get call status (auth required)
DELETE /v1/calls/{call_id}  -- Hang up a call (auth required)
GET    /health              -- Unchanged, no auth
```

**POST /v1/calls** request body (all fields except `phone_number` optional with sensible defaults):
```json
{
  "phone_number": "+1234567890",
  "system_prompt": "You are...",
  "first_message": "Hello!",
  "llm": { "model": "llama-3.3-70b-versatile", "provider": "groq", "temperature": 0.7, "max_tokens": 500 },
  "voice": { "voice_id": "abc123", "stability": 0.5, "similarity_boost": 0.75 },
  "max_duration": 300,
  "recording": true
}
```

**Response**: `201 { "call_id": "CA...", "phone_number": "+1...", "status": "queued" }`

## New Files (3)

### 1. `shuo/models.py` (~60 lines)
- `LLMConfig` / `VoiceConfig` / `CreateCallRequest` -- Pydantic request models with defaults matching current hardcoded values
- `CallConfig` -- frozen dataclass that flows through the system (API → registry → conversation → agent → services)
- `CallConfig.from_request(req)` -- factory from API request
- `DEFAULT_CONFIG` -- matches current behavior for backward compat (inbound calls, legacy endpoint)

### 2. `shuo/auth.py` (~20 lines)
- Reads `API_KEYS` env var (comma-separated)
- `require_api_key` FastAPI dependency using `HTTPBearer` -- validates Bearer token, returns 401 on failure, 500 if no keys configured

### 3. `shuo/call_registry.py` (~50 lines)
- `CallStatus` enum: `queued` / `active` / `ended`
- `CallRecord` dataclass: call_sid, config, phone_number, status, created_at
- `CallRegistry` class: `register()`, `get()`, `get_config()`, `set_active()`, `set_ended()`, `remove_old(max_age=3600)`
- Instantiated as module-level singleton

## File Modifications (8)

### 4. `shuo/types.py` -- Add `call_sid` to StreamStartEvent
```python
class StreamStartEvent:
    stream_sid: str
    call_sid: str = ""   # default preserves backward compat with existing tests
```

### 5. `shuo/services/twilio_client.py`
- Extract `callSid` from Twilio's `start.callSid` field in `parse_twilio_message`
- Add `recording: bool = True` param to `make_outbound_call`

### 6. `shuo/services/llm.py`
- Add constructor params: `system_prompt`, `model`, `provider`, `temperature`, `max_tokens` (all optional, fall back to current defaults)
- Provider-based client: `"groq"` → Groq base_url, `"openai"` → default OpenAI
- Use instance attrs in `_generate()` instead of hardcoded values / env reads
- Add `add_assistant_message(content)` for first_message support

### 7. `shuo/services/tts.py`
- Add constructor params: `voice_id`, `stability`, `similarity_boost` (all optional, fall back to env / current defaults)
- Use instance attrs in `start()` instead of hardcoded `0.5` / `0.75`

### 8. `shuo/services/tts_pool.py`
- Add voice params to `get()`: `voice_id`, `stability`, `similarity_boost`
- If requested voice matches pool defaults → dispense warm connection (existing path)
- If custom voice → create fresh TTSService with custom params (bypasses pool)
- Pass voice params through when creating fresh connections

### 9. `shuo/agent.py`
- Add `config: CallConfig = None` param to `__init__`
- Pass config fields to `LLMService` and through `TTSPool.get()`
- Add `send_first_message(message)` method: sends text directly to TTS (no LLM), adds to LLM history for context continuity

### 10. `shuo/conversation.py`
- Add `call_registry` param to `run_conversation_over_twilio()`
- On `StreamStartEvent`: extract `call_sid`, look up config from registry (fall back to `DEFAULT_CONFIG`), mark call active
- Pass `config` to `Agent.__init__`
- After Agent creation: if `config.first_message`, call `agent.send_first_message()`
- If `config.max_duration > 0`: schedule `asyncio.create_task` that sleeps then pushes `StreamStopEvent`
- In finally block: mark call ended in registry

### 11. `shuo/server.py`
- Import auth, models, registry
- Create module-level `call_registry = CallRegistry()`
- Add `POST /v1/calls` with `Depends(require_api_key)` -- validates request, calls `make_outbound_call(recording=config.recording)`, registers in registry, returns 201
- Add `GET /v1/calls/{call_id}` with auth -- returns status from registry
- Add `DELETE /v1/calls/{call_id}` with auth -- uses Twilio API to end call
- Modify `/twiml` to check recording config: Twilio POSTs with `CallSid` form field, look up in registry, conditionally include `record` attribute
- Pass `call_registry` to `run_conversation_over_twilio()` in WebSocket handler
- Keep existing `GET /call/{phone_number}` for backward compat

## Config Flow

```
POST /v1/calls (Pydantic validates body)
  → CallConfig.from_request(req)
  → make_outbound_call(to, recording=config.recording) → call_sid
  → call_registry.register(call_sid, phone, config)
  ... Twilio calls /twiml (extracts CallSid, looks up recording config) ...
  ... Twilio connects /ws ...
  → conversation: StreamStartEvent.call_sid → registry.get_config(call_sid) → config
  → Agent(config=config) → LLMService(system_prompt, model, ...) + TTSPool.get(voice_id, ...)
```

## Implementation Order

1. `shuo/models.py` (new, no deps)
2. `shuo/auth.py` (new, no deps)
3. `shuo/call_registry.py` (new, depends on models)
4. `shuo/types.py` (add call_sid field)
5. `shuo/services/twilio_client.py` (extract callSid, add recording param)
6. `shuo/services/llm.py` (configurable constructor)
7. `shuo/services/tts.py` (configurable constructor)
8. `shuo/services/tts_pool.py` (pass voice params through get())
9. `shuo/agent.py` (accept config, add send_first_message)
10. `shuo/conversation.py` (registry integration, first_message, max_duration)
11. `shuo/server.py` (new endpoints, registry wiring, twiml recording)

## Env Var

One new env var: `API_KEYS=key1,key2,key3` (comma-separated Bearer tokens)

## Verification

1. Run existing tests: `python -m pytest tests/ -v` -- should pass (StreamStartEvent backward compat)
2. Set `API_KEYS=test-key-123` in .env
3. Test auth: `curl -X POST localhost:3040/v1/calls -H "Authorization: Bearer wrong"` → 401
4. Test call: `curl -X POST localhost:3040/v1/calls -H "Authorization: Bearer test-key-123" -H "Content-Type: application/json" -d '{"phone_number": "+1234567890", "system_prompt": "You are a pirate."}'` → 201
5. Test status: `curl localhost:3040/v1/calls/CA... -H "Authorization: Bearer test-key-123"` → call status
6. Test defaults: POST with only `phone_number` → uses all default config values
