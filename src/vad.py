"""
Voice Activity Detection using Silero VAD.

Pure functions for speech detection with configurable thresholds.
No classes - just functions that transform state.
"""

import time
from typing import Tuple, Optional

import torch
import numpy as np

from .types import VADState


# =============================================================================
# CONFIGURATION (constants, not class attributes)
# =============================================================================

THRESHOLD = 0.5          # Speech probability cutoff
START_PATIENCE = 0.25    # 250ms of speech before confirming
END_PATIENCE = 0.7       # 700ms of silence before end-of-turn
WINDOW_SIZE = 512        # Silero VAD window (32ms at 16kHz)
SAMPLE_RATE = 16000      # Silero expects 16kHz


# =============================================================================
# MODEL LOADING
# =============================================================================

_model = None
_utils = None


def get_vad_model():
    """Load Silero VAD model (cached)."""
    global _model, _utils
    if _model is None:
        _model, _utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
    return _model


def reset_model_state():
    """Reset the VAD model's internal state."""
    model = get_vad_model()
    model.reset_states()


# =============================================================================
# PURE FUNCTIONS
# =============================================================================

def process_audio(
    vad_state: VADState,
    audio_16k: np.ndarray,
) -> Tuple[VADState, bool, bool]:
    """
    Process audio through VAD.
    
    Pure function: (VADState, audio) -> (VADState, speech_started, speech_ended)
    
    Args:
        vad_state: Current VAD state
        audio_16k: Audio samples at 16kHz as numpy array
        
    Returns:
        Tuple of (new_state, speech_started, speech_ended)
    """
    model = get_vad_model()
    
    # Accumulate audio in buffer
    buffer = vad_state.audio_buffer + tuple(audio_16k.tolist())
    
    # Track state changes
    is_speaking = vad_state.is_speaking
    speech_start_time = vad_state.speech_start_time
    silence_start_time = vad_state.silence_start_time
    speech_started = False
    speech_ended = False
    
    # Process complete windows
    while len(buffer) >= WINDOW_SIZE:
        window = np.array(buffer[:WINDOW_SIZE], dtype=np.float32)
        buffer = buffer[WINDOW_SIZE:]
        
        # Get speech probability
        prob = _get_speech_probability(model, window)
        
        # Update state
        (is_speaking, speech_start_time, silence_start_time, 
         started, ended) = _update_detection_state(
            is_speaking, prob, speech_start_time, silence_start_time
        )
        
        # Track if any transition happened
        if started:
            speech_started = True
        if ended:
            speech_ended = True
    
    new_state = VADState(
        is_speaking=is_speaking,
        speech_start_time=speech_start_time,
        silence_start_time=silence_start_time,
        audio_buffer=buffer
    )
    
    return new_state, speech_started, speech_ended


def _get_speech_probability(model, audio: np.ndarray) -> float:
    """Run Silero VAD on audio window."""
    audio_tensor = torch.from_numpy(audio).float()
    with torch.no_grad():
        prob = model(audio_tensor, SAMPLE_RATE).item()
    return prob


def _update_detection_state(
    is_speaking: bool,
    probability: float,
    speech_start_time: Optional[float],
    silence_start_time: Optional[float],
) -> Tuple[bool, Optional[float], Optional[float], bool, bool]:
    """
    Update detection state based on speech probability.
    
    Pure function implementing hysteresis with patience thresholds.
    
    Returns:
        Tuple of (is_speaking, speech_start_time, silence_start_time, 
                  speech_started, speech_ended)
    """
    now = time.time()
    is_speech = probability > THRESHOLD
    speech_started = False
    speech_ended = False
    
    if not is_speaking:
        # Currently in silence, check for speech start
        if is_speech:
            if speech_start_time is None:
                speech_start_time = now
            elif now - speech_start_time >= START_PATIENCE:
                is_speaking = True
                speech_started = True
                speech_start_time = None
                silence_start_time = None
        else:
            speech_start_time = None
    else:
        # Currently speaking, check for speech end
        if not is_speech:
            if silence_start_time is None:
                silence_start_time = now
            elif now - silence_start_time >= END_PATIENCE:
                is_speaking = False
                speech_ended = True
                silence_start_time = None
                speech_start_time = None
        else:
            silence_start_time = None
    
    return is_speaking, speech_start_time, silence_start_time, speech_started, speech_ended
