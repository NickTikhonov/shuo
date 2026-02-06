"""
Unit tests for the pure update function.

These tests verify the state machine logic without any I/O.
With Deepgram Flux handling turn detection, the update function
is a simple conversation controller.
"""

import pytest
from dataclasses import replace

from shuo.types import (
    AppState, Phase,
    StreamStartEvent, StreamStopEvent, MediaEvent,
    FluxStartOfTurnEvent, FluxEndOfTurnEvent, AgentTurnDoneEvent,
    FeedFluxAction, StartAgentTurnAction, ResetAgentTurnAction,
)
from shuo.update import update


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def initial_state() -> AppState:
    """Fresh state at the start of a call."""
    return AppState()


@pytest.fixture
def listening_state() -> AppState:
    """State after stream has started."""
    return AppState(phase=Phase.LISTENING, stream_sid="test-stream-sid")


@pytest.fixture
def responding_state() -> AppState:
    """State while agent is responding."""
    return AppState(phase=Phase.RESPONDING, stream_sid="test-stream-sid")


# =============================================================================
# STREAM LIFECYCLE
# =============================================================================

class TestStreamLifecycle:

    def test_stream_start_sets_stream_sid(self, initial_state):
        """StreamStartEvent should set the stream_sid."""
        event = StreamStartEvent(stream_sid="new-stream-123")
        new_state, actions = update(initial_state, event)

        assert new_state.stream_sid == "new-stream-123"
        assert new_state.phase == Phase.LISTENING
        assert actions == []

    def test_stream_start_resets_phase(self):
        """StreamStartEvent should reset to LISTENING even if RESPONDING."""
        state = AppState(phase=Phase.RESPONDING, stream_sid="old")
        event = StreamStartEvent(stream_sid="new")
        new_state, _ = update(state, event)

        assert new_state.phase == Phase.LISTENING
        assert new_state.stream_sid == "new"

    def test_stream_stop_resets_agent_if_responding(self, responding_state):
        """StreamStopEvent should reset agent turn if responding."""
        event = StreamStopEvent()
        _, actions = update(responding_state, event)

        assert len(actions) == 1
        assert isinstance(actions[0], ResetAgentTurnAction)

    def test_stream_stop_no_action_if_listening(self, listening_state):
        """StreamStopEvent should produce no actions when listening."""
        event = StreamStopEvent()
        _, actions = update(listening_state, event)

        assert actions == []


# =============================================================================
# MEDIA ROUTING
# =============================================================================

class TestMediaRouting:

    def test_media_feeds_flux(self, listening_state):
        """MediaEvent should always produce FeedFluxAction."""
        event = MediaEvent(audio_bytes=b"\x00\x01\x02")
        _, actions = update(listening_state, event)

        assert len(actions) == 1
        assert isinstance(actions[0], FeedFluxAction)
        assert actions[0].audio_bytes == b"\x00\x01\x02"

    def test_media_feeds_flux_in_any_phase(self, responding_state):
        """Audio should flow to Flux regardless of phase."""
        event = MediaEvent(audio_bytes=b"\xff")
        _, actions = update(responding_state, event)

        assert len(actions) == 1
        assert isinstance(actions[0], FeedFluxAction)

    def test_media_does_not_change_state(self, listening_state):
        """MediaEvent should not change application state."""
        event = MediaEvent(audio_bytes=b"\x00")
        new_state, _ = update(listening_state, event)

        assert new_state == listening_state


# =============================================================================
# FLUX TURN EVENTS
# =============================================================================

class TestFluxEndOfTurn:

    def test_end_of_turn_starts_agent(self, listening_state):
        """FluxEndOfTurnEvent with transcript should start agent turn."""
        event = FluxEndOfTurnEvent(transcript="Hello, how are you?")
        new_state, actions = update(listening_state, event)

        assert new_state.phase == Phase.RESPONDING
        assert len(actions) == 1
        assert isinstance(actions[0], StartAgentTurnAction)
        assert actions[0].transcript == "Hello, how are you?"

    def test_end_of_turn_empty_transcript_ignored(self, listening_state):
        """Empty transcript should be ignored."""
        event = FluxEndOfTurnEvent(transcript="")
        new_state, actions = update(listening_state, event)

        assert new_state.phase == Phase.LISTENING
        assert actions == []

    def test_end_of_turn_ignored_if_responding(self, responding_state):
        """EndOfTurn should be ignored if already responding."""
        event = FluxEndOfTurnEvent(transcript="Interrupt text")
        new_state, actions = update(responding_state, event)

        assert new_state.phase == Phase.RESPONDING  # Unchanged
        assert actions == []


class TestFluxStartOfTurn:

    def test_start_of_turn_interrupts_agent(self, responding_state):
        """StartOfTurn during RESPONDING should trigger barge-in."""
        event = FluxStartOfTurnEvent()
        new_state, actions = update(responding_state, event)

        assert new_state.phase == Phase.LISTENING
        assert len(actions) == 1
        assert isinstance(actions[0], ResetAgentTurnAction)

    def test_start_of_turn_ignored_if_listening(self, listening_state):
        """StartOfTurn during LISTENING should be a no-op."""
        event = FluxStartOfTurnEvent()
        new_state, actions = update(listening_state, event)

        assert new_state.phase == Phase.LISTENING
        assert actions == []


# =============================================================================
# AGENT TURN DONE
# =============================================================================

class TestAgentTurnDone:

    def test_done_transitions_to_listening(self, responding_state):
        """AgentTurnDoneEvent should transition back to LISTENING."""
        event = AgentTurnDoneEvent()
        new_state, actions = update(responding_state, event)

        assert new_state.phase == Phase.LISTENING
        assert actions == []

    def test_done_ignored_if_listening(self, listening_state):
        """AgentTurnDoneEvent should be ignored if already listening."""
        event = AgentTurnDoneEvent()
        new_state, actions = update(listening_state, event)

        assert new_state.phase == Phase.LISTENING
        assert actions == []


# =============================================================================
# COMPLETE FLOW
# =============================================================================

class TestCompleteFlow:

    def test_full_conversation_turn(self, listening_state):
        """Complete turn: Flux EndOfTurn -> Agent responds -> Done."""
        state = listening_state

        # Flux detects end of user turn
        state, actions = update(state, FluxEndOfTurnEvent(transcript="Hello"))
        assert state.phase == Phase.RESPONDING
        assert any(isinstance(a, StartAgentTurnAction) for a in actions)

        # Agent finishes speaking
        state, actions = update(state, AgentTurnDoneEvent())
        assert state.phase == Phase.LISTENING
        assert actions == []

    def test_interrupt_during_response(self, listening_state):
        """Barge-in: Agent responding -> Flux StartOfTurn -> Reset."""
        state = listening_state

        # Start responding
        state, _ = update(state, FluxEndOfTurnEvent(transcript="Hello"))
        assert state.phase == Phase.RESPONDING

        # User interrupts
        state, actions = update(state, FluxStartOfTurnEvent())
        assert state.phase == Phase.LISTENING
        assert any(isinstance(a, ResetAgentTurnAction) for a in actions)

    def test_multi_turn(self, listening_state):
        """Multiple turns work correctly."""
        state = listening_state

        # Turn 1
        state, _ = update(state, FluxEndOfTurnEvent(transcript="Hi"))
        assert state.phase == Phase.RESPONDING
        state, _ = update(state, AgentTurnDoneEvent())
        assert state.phase == Phase.LISTENING

        # Turn 2
        state, _ = update(state, FluxEndOfTurnEvent(transcript="How are you?"))
        assert state.phase == Phase.RESPONDING
        state, _ = update(state, AgentTurnDoneEvent())
        assert state.phase == Phase.LISTENING

    def test_audio_always_forwarded_to_flux(self, listening_state):
        """MediaEvents should always produce FeedFluxAction."""
        state = listening_state

        # While listening
        _, actions = update(state, MediaEvent(audio_bytes=b"\x00"))
        assert isinstance(actions[0], FeedFluxAction)

        # While responding
        state = replace(state, phase=Phase.RESPONDING)
        _, actions = update(state, MediaEvent(audio_bytes=b"\x00"))
        assert isinstance(actions[0], FeedFluxAction)

    def test_interrupt_then_new_turn(self, listening_state):
        """After barge-in, a new turn can start."""
        state = listening_state

        # Start turn
        state, _ = update(state, FluxEndOfTurnEvent(transcript="Hello"))
        assert state.phase == Phase.RESPONDING

        # Interrupt
        state, actions = update(state, FluxStartOfTurnEvent())
        assert state.phase == Phase.LISTENING
        assert any(isinstance(a, ResetAgentTurnAction) for a in actions)

        # New turn after interrupt
        state, actions = update(state, FluxEndOfTurnEvent(transcript="Never mind, goodbye"))
        assert state.phase == Phase.RESPONDING
        assert any(isinstance(a, StartAgentTurnAction) for a in actions)
        assert actions[0].transcript == "Never mind, goodbye"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:

    def test_state_immutability(self, initial_state):
        """State updates should not mutate original."""
        event = StreamStartEvent(stream_sid="new-sid")
        new_state, _ = update(initial_state, event)

        assert initial_state.stream_sid is None
        assert new_state.stream_sid == "new-sid"

    def test_stream_stop_in_listening_is_safe(self, listening_state):
        """StreamStopEvent while listening should produce no actions."""
        _, actions = update(listening_state, StreamStopEvent())
        assert actions == []

    def test_agent_done_in_wrong_phase_is_safe(self, listening_state):
        """AgentTurnDoneEvent in LISTENING should not crash."""
        new_state, actions = update(listening_state, AgentTurnDoneEvent())
        assert new_state.phase == Phase.LISTENING
        assert actions == []

    def test_start_of_turn_in_listening_is_safe(self, listening_state):
        """FluxStartOfTurnEvent in LISTENING is normal (user talking)."""
        new_state, actions = update(listening_state, FluxStartOfTurnEvent())
        assert new_state.phase == Phase.LISTENING
        assert actions == []

    def test_end_of_turn_without_stream_sid(self, initial_state):
        """EndOfTurn before stream starts should still work."""
        new_state, actions = update(initial_state, FluxEndOfTurnEvent(transcript="test"))
        assert new_state.phase == Phase.RESPONDING
        assert len(actions) == 1
