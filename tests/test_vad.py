"""
Unit tests for the VAD (Voice Activity Detection) state machine.

Tests the pure _update_detection_state function which implements
the hysteresis logic for speech detection.
"""

import pytest
import time
from unittest.mock import patch

from src.types import VADState
from src.vad import (
    _update_detection_state,
    THRESHOLD,
    START_PATIENCE,
    END_PATIENCE,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fixed_time():
    """Fixture to control time in tests."""
    with patch('src.vad.time.time') as mock_time:
        mock_time.return_value = 1000.0  # Arbitrary start time
        yield mock_time


# =============================================================================
# BASIC DETECTION TESTS
# =============================================================================

class TestBasicDetection:
    """Tests for basic speech/silence detection."""
    
    def test_silence_stays_silent_below_threshold(self, fixed_time):
        """Low probability should maintain silence state."""
        is_speaking, speech_start, silence_start, started, ended = _update_detection_state(
            is_speaking=False,
            probability=0.2,  # Below THRESHOLD
            speech_start_time=None,
            silence_start_time=None,
        )
        
        assert is_speaking == False
        assert started == False
        assert ended == False
    
    def test_high_probability_starts_timing(self, fixed_time):
        """High probability should start speech timing."""
        is_speaking, speech_start, silence_start, started, ended = _update_detection_state(
            is_speaking=False,
            probability=0.8,  # Above THRESHOLD
            speech_start_time=None,
            silence_start_time=None,
        )
        
        assert is_speaking == False  # Not yet - patience not elapsed
        assert speech_start == 1000.0  # Started timing
        assert started == False
    
    def test_speech_confirmed_after_patience(self, fixed_time):
        """Speech should be confirmed after START_PATIENCE elapsed."""
        # Simulate time passing beyond patience
        past_time = 1000.0 - START_PATIENCE - 0.1  # Started enough time ago
        
        is_speaking, speech_start, silence_start, started, ended = _update_detection_state(
            is_speaking=False,
            probability=0.8,
            speech_start_time=past_time,  # Started in the past
            silence_start_time=None,
        )
        
        assert is_speaking == True
        assert started == True
        assert speech_start is None  # Reset after confirmation
    
    def test_speech_timing_reset_on_silence(self, fixed_time):
        """Speech timing should reset if silence detected before patience."""
        is_speaking, speech_start, silence_start, started, ended = _update_detection_state(
            is_speaking=False,
            probability=0.2,  # Below threshold
            speech_start_time=999.9,  # Was timing
            silence_start_time=None,
        )
        
        assert speech_start is None  # Reset
        assert started == False


# =============================================================================
# SPEECH END DETECTION TESTS
# =============================================================================

class TestSpeechEndDetection:
    """Tests for detecting when speech ends."""
    
    def test_low_probability_during_speech_starts_timing(self, fixed_time):
        """Low probability during speech should start silence timing."""
        is_speaking, speech_start, silence_start, started, ended = _update_detection_state(
            is_speaking=True,
            probability=0.2,  # Below threshold
            speech_start_time=None,
            silence_start_time=None,
        )
        
        assert is_speaking == True  # Not ended yet
        assert silence_start == 1000.0  # Started timing silence
        assert ended == False
    
    def test_speech_ends_after_silence_patience(self, fixed_time):
        """Speech should end after END_PATIENCE of silence."""
        past_time = 1000.0 - END_PATIENCE - 0.1  # Silence started long enough ago
        
        is_speaking, speech_start, silence_start, started, ended = _update_detection_state(
            is_speaking=True,
            probability=0.2,
            speech_start_time=None,
            silence_start_time=past_time,
        )
        
        assert is_speaking == False
        assert ended == True
        assert silence_start is None  # Reset
    
    def test_silence_timing_reset_on_speech(self, fixed_time):
        """Silence timing should reset if speech detected again."""
        is_speaking, speech_start, silence_start, started, ended = _update_detection_state(
            is_speaking=True,
            probability=0.8,  # Above threshold
            speech_start_time=None,
            silence_start_time=999.9,  # Was timing
        )
        
        assert is_speaking == True  # Still speaking
        assert silence_start is None  # Reset
        assert ended == False


# =============================================================================
# HYSTERESIS TESTS
# =============================================================================

class TestHysteresis:
    """Tests for hysteresis behavior (prevents flickering)."""
    
    def test_brief_speech_not_detected(self, fixed_time):
        """Brief speech (< START_PATIENCE) should not trigger detection."""
        # First frame: speech detected, start timing
        is_speaking, speech_start, _, _, _ = _update_detection_state(
            is_speaking=False,
            probability=0.8,
            speech_start_time=None,
            silence_start_time=None,
        )
        assert speech_start == 1000.0
        
        # Advance time slightly (not enough for patience)
        fixed_time.return_value = 1000.0 + START_PATIENCE / 2
        
        # Still high probability but not enough time
        is_speaking, speech_start, _, started, _ = _update_detection_state(
            is_speaking=False,
            probability=0.8,
            speech_start_time=1000.0,
            silence_start_time=None,
        )
        assert is_speaking == False
        assert started == False
        
        # Then silence - timing should reset
        is_speaking, speech_start, _, started, _ = _update_detection_state(
            is_speaking=False,
            probability=0.2,
            speech_start_time=1000.0,
            silence_start_time=None,
        )
        assert speech_start is None
    
    def test_brief_silence_not_end_of_speech(self, fixed_time):
        """Brief silence (< END_PATIENCE) should not end speech."""
        # Speaking, then brief silence
        is_speaking, _, silence_start, _, _ = _update_detection_state(
            is_speaking=True,
            probability=0.2,
            speech_start_time=None,
            silence_start_time=None,
        )
        assert silence_start == 1000.0
        
        # Advance time slightly
        fixed_time.return_value = 1000.0 + END_PATIENCE / 2
        
        # Resume speaking before patience elapsed
        is_speaking, _, silence_start, _, ended = _update_detection_state(
            is_speaking=True,
            probability=0.8,
            speech_start_time=None,
            silence_start_time=1000.0,
        )
        assert is_speaking == True
        assert silence_start is None
        assert ended == False


# =============================================================================
# THRESHOLD TESTS  
# =============================================================================

class TestThreshold:
    """Tests for threshold boundary behavior."""
    
    def test_exactly_at_threshold_is_not_speech(self, fixed_time):
        """Probability exactly at threshold should not be speech."""
        is_speaking, speech_start, _, _, _ = _update_detection_state(
            is_speaking=False,
            probability=THRESHOLD,  # Exactly at threshold
            speech_start_time=None,
            silence_start_time=None,
        )
        
        # > THRESHOLD, not >= THRESHOLD
        assert speech_start is None
    
    def test_just_above_threshold_is_speech(self, fixed_time):
        """Probability just above threshold should be speech."""
        is_speaking, speech_start, _, _, _ = _update_detection_state(
            is_speaking=False,
            probability=THRESHOLD + 0.01,
            speech_start_time=None,
            silence_start_time=None,
        )
        
        assert speech_start == 1000.0


# =============================================================================
# STATE MACHINE CONSISTENCY TESTS
# =============================================================================

class TestStateMachineConsistency:
    """Tests for overall state machine consistency."""
    
    def test_cannot_start_and_end_simultaneously(self, fixed_time):
        """Speech cannot both start and end in the same frame."""
        # This is impossible by the state machine design
        # If we're not speaking, we can only start
        # If we are speaking, we can only end
        
        # Starting from silence
        _, _, _, started, ended = _update_detection_state(
            is_speaking=False,
            probability=0.8,
            speech_start_time=1000.0 - START_PATIENCE - 0.1,
            silence_start_time=None,
        )
        assert started == True
        assert ended == False
        
        # Starting from speech
        _, _, _, started, ended = _update_detection_state(
            is_speaking=True,
            probability=0.2,
            speech_start_time=None,
            silence_start_time=1000.0 - END_PATIENCE - 0.1,
        )
        assert started == False
        assert ended == True
    
    def test_state_transitions_are_clean(self, fixed_time):
        """State transitions should reset timing cleanly."""
        # Transition from silence to speech
        is_speaking, speech_start, silence_start, _, _ = _update_detection_state(
            is_speaking=False,
            probability=0.8,
            speech_start_time=1000.0 - START_PATIENCE - 0.1,
            silence_start_time=None,
        )
        
        assert is_speaking == True
        assert speech_start is None
        assert silence_start is None
        
        # Transition from speech to silence
        fixed_time.return_value = 2000.0
        is_speaking, speech_start, silence_start, _, _ = _update_detection_state(
            is_speaking=True,
            probability=0.2,
            speech_start_time=None,
            silence_start_time=2000.0 - END_PATIENCE - 0.1,
        )
        
        assert is_speaking == False
        assert speech_start is None
        assert silence_start is None


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

class TestParameterValidation:
    """Tests that verify our patience parameters are reasonable."""
    
    def test_start_patience_is_positive(self):
        """START_PATIENCE should be positive."""
        assert START_PATIENCE > 0
    
    def test_end_patience_is_positive(self):
        """END_PATIENCE should be positive."""
        assert END_PATIENCE > 0
    
    def test_end_patience_longer_than_start(self):
        """END_PATIENCE should be longer than START_PATIENCE for natural conversation."""
        # We want to detect speech start quickly
        # but wait longer before assuming speech ended
        assert END_PATIENCE >= START_PATIENCE
    
    def test_threshold_is_between_0_and_1(self):
        """Threshold should be a valid probability."""
        assert 0 < THRESHOLD < 1
