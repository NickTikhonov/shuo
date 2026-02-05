"""
Unit tests for the pure update function.

These tests verify the state machine logic without any I/O.
The update function is pure: (State, Event) -> (State, Actions)
"""

import pytest
from dataclasses import replace

from src.types import (
    # State
    AppState, Phase, VADState, Message,
    # Events
    StreamStartEvent, StreamStopEvent, MediaEvent,
    STTPartialEvent, STTFinalEvent,
    LLMTokenEvent, LLMDoneEvent,
    TTSAudioEvent, TTSDoneEvent,
    PlaybackDoneEvent,
    # Actions
    StartSTTAction, FeedSTTAction, StopSTTAction, CancelSTTAction,
    StartLLMAction, CancelLLMAction,
    StartTTSAction, FeedTTSAction, FlushTTSAction, CancelTTSAction,
    StartPlaybackAction, StopPlaybackAction,
)
from src.update import (
    update,
    handle_stream_start,
    handle_stream_stop,
    handle_stt_final,
    handle_llm_token,
    handle_llm_done,
    handle_tts_audio,
    handle_tts_done,
    handle_playback_done,
    handle_interrupt,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def initial_state() -> AppState:
    """Fresh state at the start of a call."""
    return AppState(
        phase=Phase.LISTENING,
        vad=VADState(),
        stream_sid=None,
    )


@pytest.fixture
def listening_state() -> AppState:
    """State after stream has started, listening for user."""
    return AppState(
        phase=Phase.LISTENING,
        vad=VADState(),
        stream_sid="test-stream-sid",
    )


@pytest.fixture
def stt_active_state() -> AppState:
    """State while STT is transcribing user speech."""
    return AppState(
        phase=Phase.LISTENING,
        vad=VADState(is_speaking=True),
        stream_sid="test-stream-sid",
        stt_active=True,
        current_transcript="",
    )


@pytest.fixture
def processing_state() -> AppState:
    """State after user finished speaking, LLM generating."""
    return AppState(
        phase=Phase.PROCESSING,
        vad=VADState(),
        stream_sid="test-stream-sid",
        stt_active=False,
        current_transcript="Hello, how are you?",
        llm_active=True,
        pending_llm_text="",
        tts_active=True,
    )


@pytest.fixture
def speaking_state() -> AppState:
    """State while playing audio back to user."""
    return AppState(
        phase=Phase.SPEAKING,
        vad=VADState(),
        stream_sid="test-stream-sid",
        stt_active=False,
        current_transcript="Hello, how are you?",
        llm_active=False,
        pending_llm_text="I'm doing well, thanks!",
        tts_active=False,
    )


# =============================================================================
# STREAM LIFECYCLE TESTS
# =============================================================================

class TestStreamLifecycle:
    """Tests for stream start/stop handling."""
    
    def test_stream_start_sets_stream_sid(self, initial_state):
        """StreamStartEvent should set the stream_sid."""
        event = StreamStartEvent(stream_sid="new-stream-123")
        new_state, actions = update(initial_state, event)
        
        assert new_state.stream_sid == "new-stream-123"
        assert new_state.phase == Phase.LISTENING
        assert actions == []
    
    def test_stream_start_resets_state(self):
        """StreamStartEvent should reset any existing state."""
        dirty_state = AppState(
            phase=Phase.SPEAKING,
            stream_sid="old-stream",
            stt_active=True,
            llm_active=True,
            tts_active=True,
            current_transcript="old text",
            pending_llm_text="old response",
        )
        
        event = StreamStartEvent(stream_sid="new-stream")
        new_state, actions = update(dirty_state, event)
        
        assert new_state.stream_sid == "new-stream"
        assert new_state.phase == Phase.LISTENING
        assert new_state.stt_active == False
        assert new_state.llm_active == False
        assert new_state.tts_active == False
        assert new_state.current_transcript == ""
        assert new_state.pending_llm_text == ""
    
    def test_stream_stop_cancels_active_services(self):
        """StreamStopEvent should cancel all active services."""
        state = AppState(
            phase=Phase.PROCESSING,
            stream_sid="test",
            stt_active=True,
            llm_active=True,
            tts_active=True,
        )
        
        event = StreamStopEvent()
        new_state, actions = update(state, event)
        
        # Should have cancel actions for all active services
        action_types = [type(a) for a in actions]
        assert CancelSTTAction in action_types
        assert CancelLLMAction in action_types
        assert CancelTTSAction in action_types
    
    def test_stream_stop_stops_playback_if_speaking(self, speaking_state):
        """StreamStopEvent should stop playback if in SPEAKING phase."""
        event = StreamStopEvent()
        new_state, actions = update(speaking_state, event)
        
        action_types = [type(a) for a in actions]
        assert StopPlaybackAction in action_types


# =============================================================================
# STT FLOW TESTS
# =============================================================================

class TestSTTFlow:
    """Tests for Speech-to-Text flow."""
    
    def test_stt_partial_does_not_change_state(self, stt_active_state):
        """STTPartialEvent should not change state (just for logging)."""
        event = STTPartialEvent(text="Hello")
        new_state, actions = update(stt_active_state, event)
        
        assert new_state == stt_active_state
        assert actions == []
    
    def test_stt_final_transitions_to_processing(self, listening_state):
        """STTFinalEvent should transition to PROCESSING and start LLM+TTS."""
        state = replace(listening_state, stt_active=True)
        event = STTFinalEvent(text="Hello, how are you?")
        
        new_state, actions = update(state, event)
        
        assert new_state.phase == Phase.PROCESSING
        assert new_state.stt_active == False
        assert new_state.current_transcript == "Hello, how are you?"
        assert new_state.llm_active == True
        assert new_state.tts_active == True
        
        action_types = [type(a) for a in actions]
        assert StartLLMAction in action_types
        assert StartTTSAction in action_types
    
    def test_stt_final_starts_llm_with_correct_message(self, listening_state):
        """STTFinalEvent should start LLM with the transcribed message."""
        state = replace(listening_state, stt_active=True)
        event = STTFinalEvent(text="What is the weather?")
        
        new_state, actions = update(state, event)
        
        # Find the StartLLMAction
        llm_action = next(a for a in actions if isinstance(a, StartLLMAction))
        assert llm_action.user_message == "What is the weather?"
    
    def test_stt_final_empty_text_ignored(self, listening_state):
        """Empty STT transcript should be ignored."""
        state = replace(listening_state, stt_active=True)
        event = STTFinalEvent(text="   ")  # Whitespace only
        
        new_state, actions = update(state, event)
        
        assert new_state.phase == Phase.LISTENING  # No transition
        assert new_state.stt_active == False
        assert actions == []
    
    def test_stt_final_preserves_conversation_history(self, listening_state):
        """STTFinalEvent should use existing conversation history."""
        history = (
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
        )
        state = replace(
            listening_state,
            stt_active=True,
            conversation_history=history,
        )
        event = STTFinalEvent(text="How are you?")
        
        new_state, actions = update(state, event)
        
        llm_action = next(a for a in actions if isinstance(a, StartLLMAction))
        assert llm_action.history == history


# =============================================================================
# LLM FLOW TESTS
# =============================================================================

class TestLLMFlow:
    """Tests for LLM generation flow."""
    
    def test_llm_token_forwards_to_tts(self, processing_state):
        """LLMTokenEvent should forward token to TTS."""
        event = LLMTokenEvent(token="Hello")
        new_state, actions = update(processing_state, event)
        
        assert new_state.pending_llm_text == "Hello"
        assert len(actions) == 1
        assert isinstance(actions[0], FeedTTSAction)
        assert actions[0].text == "Hello"
    
    def test_llm_tokens_accumulate(self, processing_state):
        """Multiple LLM tokens should accumulate."""
        state = processing_state
        
        state, _ = update(state, LLMTokenEvent(token="I "))
        state, _ = update(state, LLMTokenEvent(token="am "))
        state, _ = update(state, LLMTokenEvent(token="fine."))
        
        assert state.pending_llm_text == "I am fine."
    
    def test_llm_token_ignored_if_tts_inactive(self, processing_state):
        """LLM tokens should be ignored if TTS is not active."""
        state = replace(processing_state, tts_active=False)
        event = LLMTokenEvent(token="Hello")
        
        new_state, actions = update(state, event)
        
        assert actions == []
    
    def test_llm_done_flushes_tts(self, processing_state):
        """LLMDoneEvent should flush TTS buffer."""
        state = replace(
            processing_state,
            pending_llm_text="Complete response.",
            current_transcript="User question",
        )
        event = LLMDoneEvent()
        
        new_state, actions = update(state, event)
        
        assert new_state.llm_active == False
        action_types = [type(a) for a in actions]
        assert FlushTTSAction in action_types
    
    def test_llm_done_updates_conversation_history(self, processing_state):
        """LLMDoneEvent should add messages to history."""
        state = replace(
            processing_state,
            current_transcript="What time is it?",
            pending_llm_text="It's 3pm.",
        )
        event = LLMDoneEvent()
        
        new_state, actions = update(state, event)
        
        assert len(new_state.conversation_history) == 2
        assert new_state.conversation_history[0] == Message(role="user", content="What time is it?")
        assert new_state.conversation_history[1] == Message(role="assistant", content="It's 3pm.")


# =============================================================================
# TTS FLOW TESTS
# =============================================================================

class TestTTSFlow:
    """Tests for Text-to-Speech flow."""
    
    def test_tts_audio_transitions_to_speaking(self, processing_state):
        """First TTSAudioEvent should transition to SPEAKING."""
        event = TTSAudioEvent(audio_base64="base64audiodata")
        new_state, actions = update(processing_state, event)
        
        assert new_state.phase == Phase.SPEAKING
        action_types = [type(a) for a in actions]
        assert StartPlaybackAction in action_types
    
    def test_tts_audio_no_transition_if_already_speaking(self, speaking_state):
        """TTSAudioEvent should not re-transition if already SPEAKING."""
        event = TTSAudioEvent(audio_base64="more audio")
        new_state, actions = update(speaking_state, event)
        
        assert new_state.phase == Phase.SPEAKING
        # Should not have StartPlaybackAction again
        action_types = [type(a) for a in actions]
        assert StartPlaybackAction not in action_types
    
    def test_tts_done_marks_tts_inactive(self, speaking_state):
        """TTSDoneEvent should mark TTS as inactive."""
        state = replace(speaking_state, tts_active=True)
        event = TTSDoneEvent()
        
        new_state, actions = update(state, event)
        
        assert new_state.tts_active == False


# =============================================================================
# PLAYBACK TESTS
# =============================================================================

class TestPlayback:
    """Tests for playback completion handling."""
    
    def test_playback_done_transitions_to_listening(self, speaking_state):
        """PlaybackDoneEvent should transition back to LISTENING."""
        event = PlaybackDoneEvent()
        new_state, actions = update(speaking_state, event)
        
        assert new_state.phase == Phase.LISTENING
    
    def test_playback_done_ignored_if_not_speaking(self, listening_state):
        """PlaybackDoneEvent should be ignored if not in SPEAKING phase."""
        event = PlaybackDoneEvent()
        new_state, actions = update(listening_state, event)
        
        assert new_state.phase == Phase.LISTENING  # Unchanged
        assert actions == []
    
    def test_playback_done_ignored_during_processing(self, processing_state):
        """PlaybackDoneEvent should be ignored during PROCESSING."""
        event = PlaybackDoneEvent()
        new_state, actions = update(processing_state, event)
        
        assert new_state.phase == Phase.PROCESSING  # Unchanged


# =============================================================================
# INTERRUPT TESTS
# =============================================================================

class TestInterrupt:
    """Tests for interrupt handling (barge-in)."""
    
    def test_interrupt_cancels_llm(self):
        """Interrupt should cancel LLM if active."""
        state = AppState(
            phase=Phase.PROCESSING,
            stream_sid="test",
            llm_active=True,
        )
        
        new_state, actions = handle_interrupt(state)
        
        assert new_state.llm_active == False
        action_types = [type(a) for a in actions]
        assert CancelLLMAction in action_types
    
    def test_interrupt_cancels_tts(self):
        """Interrupt should cancel TTS if active."""
        state = AppState(
            phase=Phase.PROCESSING,
            stream_sid="test",
            tts_active=True,
        )
        
        new_state, actions = handle_interrupt(state)
        
        assert new_state.tts_active == False
        action_types = [type(a) for a in actions]
        assert CancelTTSAction in action_types
    
    def test_interrupt_stops_playback(self, speaking_state):
        """Interrupt should always stop playback."""
        new_state, actions = handle_interrupt(speaking_state)
        
        action_types = [type(a) for a in actions]
        assert StopPlaybackAction in action_types
    
    def test_interrupt_starts_fresh_stt(self, speaking_state):
        """Interrupt should start fresh STT session."""
        new_state, actions = handle_interrupt(speaking_state)
        
        assert new_state.stt_active == True
        action_types = [type(a) for a in actions]
        assert StartSTTAction in action_types
    
    def test_interrupt_transitions_to_listening(self, speaking_state):
        """Interrupt should transition back to LISTENING."""
        new_state, actions = handle_interrupt(speaking_state)
        
        assert new_state.phase == Phase.LISTENING
    
    def test_interrupt_clears_pending_text(self):
        """Interrupt should clear any pending LLM text."""
        state = AppState(
            phase=Phase.PROCESSING,
            stream_sid="test",
            llm_active=True,
            pending_llm_text="Partial response...",
            current_transcript="User question",
        )
        
        new_state, actions = handle_interrupt(state)
        
        assert new_state.pending_llm_text == ""
        assert new_state.current_transcript == ""


# =============================================================================
# COMPLETE FLOW TESTS
# =============================================================================

class TestCompleteFlow:
    """End-to-end tests of the conversation flow."""
    
    def test_full_turn_flow(self, listening_state):
        """Test a complete user turn: speak -> transcribe -> generate -> play."""
        state = listening_state
        
        # User speaks (simulated - normally VAD would set this)
        state = replace(state, stt_active=True)
        
        # STT returns final transcript
        state, actions = update(state, STTFinalEvent(text="Hello"))
        assert state.phase == Phase.PROCESSING
        assert any(isinstance(a, StartLLMAction) for a in actions)
        assert any(isinstance(a, StartTTSAction) for a in actions)
        
        # LLM streams tokens
        state, actions = update(state, LLMTokenEvent(token="Hi "))
        assert any(isinstance(a, FeedTTSAction) for a in actions)
        
        state, actions = update(state, LLMTokenEvent(token="there!"))
        assert state.pending_llm_text == "Hi there!"
        
        # LLM done
        state, actions = update(state, LLMDoneEvent())
        assert state.llm_active == False
        assert any(isinstance(a, FlushTTSAction) for a in actions)
        
        # TTS produces audio - transition to speaking
        state, actions = update(state, TTSAudioEvent(audio_base64="audio1"))
        assert state.phase == Phase.SPEAKING
        assert any(isinstance(a, StartPlaybackAction) for a in actions)
        
        # TTS done
        state, actions = update(state, TTSDoneEvent())
        assert state.tts_active == False
        
        # Playback done - back to listening
        state, actions = update(state, PlaybackDoneEvent())
        assert state.phase == Phase.LISTENING
        
        # Verify conversation history
        assert len(state.conversation_history) == 2
        assert state.conversation_history[0].content == "Hello"
        assert state.conversation_history[1].content == "Hi there!"
    
    def test_interrupt_during_speaking(self, speaking_state):
        """Test interrupt while agent is speaking."""
        state = replace(
            speaking_state,
            llm_active=False,  # LLM already done
            tts_active=True,   # But TTS still active
        )
        
        # Simulate interrupt via handle_interrupt (normally triggered by VAD)
        new_state, actions = handle_interrupt(state)
        
        # Should cancel TTS and stop playback
        action_types = [type(a) for a in actions]
        assert CancelTTSAction in action_types
        assert StopPlaybackAction in action_types
        assert StartSTTAction in action_types
        
        # Should be back in listening mode
        assert new_state.phase == Phase.LISTENING
        assert new_state.stt_active == True
    
    def test_interrupt_during_processing(self, processing_state):
        """Test interrupt while LLM/TTS are generating."""
        new_state, actions = handle_interrupt(processing_state)
        
        # Should cancel both LLM and TTS
        action_types = [type(a) for a in actions]
        assert CancelLLMAction in action_types
        assert CancelTTSAction in action_types
        assert StopPlaybackAction in action_types
        
        # Should start fresh STT
        assert StartSTTAction in action_types
        assert new_state.stt_active == True
    
    def test_multi_turn_conversation(self, listening_state):
        """Test multiple conversation turns build history correctly."""
        state = listening_state
        
        # Turn 1
        state = replace(state, stt_active=True)
        state, _ = update(state, STTFinalEvent(text="Hi"))
        state, _ = update(state, LLMTokenEvent(token="Hello!"))
        state, _ = update(state, LLMDoneEvent())
        state, _ = update(state, TTSAudioEvent(audio_base64="a"))
        state, _ = update(state, TTSDoneEvent())
        state, _ = update(state, PlaybackDoneEvent())
        
        assert len(state.conversation_history) == 2
        
        # Turn 2
        state = replace(state, stt_active=True)
        state, _ = update(state, STTFinalEvent(text="How are you?"))
        state, _ = update(state, LLMTokenEvent(token="I'm good!"))
        state, _ = update(state, LLMDoneEvent())
        state, _ = update(state, TTSAudioEvent(audio_base64="b"))
        state, _ = update(state, TTSDoneEvent())
        state, _ = update(state, PlaybackDoneEvent())
        
        # Should have 4 messages now
        assert len(state.conversation_history) == 4
        assert state.conversation_history[2].content == "How are you?"
        assert state.conversation_history[3].content == "I'm good!"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_events_in_wrong_phase_are_safe(self, listening_state):
        """Events in the wrong phase should not crash."""
        # TTS events when not processing
        state, actions = update(listening_state, TTSAudioEvent(audio_base64="x"))
        # Should just not transition
        assert state.phase == Phase.LISTENING
        
        # Playback done when not speaking
        state, actions = update(listening_state, PlaybackDoneEvent())
        assert state.phase == Phase.LISTENING
    
    def test_duplicate_stt_final_ignored(self, processing_state):
        """Duplicate STT final should not restart pipeline."""
        # Already in processing
        event = STTFinalEvent(text="Another message")
        new_state, actions = update(processing_state, event)
        
        # Should not have new LLM action since we're already processing
        # (STT is inactive in processing_state)
        assert new_state.stt_active == False
    
    def test_state_immutability(self, initial_state):
        """State updates should not mutate original."""
        original_phase = initial_state.phase
        
        event = StreamStartEvent(stream_sid="new-sid")
        new_state, _ = update(initial_state, event)
        
        # Original should be unchanged
        assert initial_state.stream_sid is None
        assert initial_state.phase == original_phase
        
        # New state has changes
        assert new_state.stream_sid == "new-sid"
