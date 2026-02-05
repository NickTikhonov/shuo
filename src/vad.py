"""
Voice Activity Detection using Silero VAD.

Provides speech state tracking with configurable thresholds for:
- Speech start detection (start_patience)
- Speech end detection (end_patience)
- Probability threshold

The VAD processes 16kHz audio and outputs speech events.
"""

import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np


class SpeechState(Enum):
    """Current state of speech detection."""
    SILENCE = auto()      # No speech detected
    SPEAKING = auto()     # User is currently speaking
    SPEECH_START = auto() # Just started speaking (transition event)
    SPEECH_END = auto()   # Just stopped speaking (transition event)


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""
    # Probability threshold for speech detection (0.0 - 1.0)
    threshold: float = 0.5
    
    # Minimum duration of speech to trigger SPEECH_START (seconds)
    # Prevents false triggers from clicks, coughs, etc.
    start_patience: float = 0.25  # 250ms
    
    # Minimum duration of silence to trigger SPEECH_END (seconds)
    # Higher = more natural pauses allowed, Lower = faster response
    end_patience: float = 0.7  # 700ms
    
    # Sample rate expected by Silero VAD
    sample_rate: int = 16000


class VoiceActivityDetector:
    """
    Silero VAD wrapper with speech state tracking.
    
    Maintains internal state to detect speech start/end transitions
    with configurable patience thresholds.
    """
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        
        # Load Silero VAD model
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        
        # Internal state
        self._is_speaking = False
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        
        # Audio buffer for accumulating samples
        self._audio_buffer = np.array([], dtype=np.float32)
        
        # Silero VAD window size (512 samples at 16kHz = 32ms)
        self._window_size = 512
    
    def reset(self) -> None:
        """Reset VAD state for a new conversation."""
        self._is_speaking = False
        self._speech_start_time = None
        self._silence_start_time = None
        self._audio_buffer = np.array([], dtype=np.float32)
        self.model.reset_states()
    
    def process(self, audio: np.ndarray) -> SpeechState:
        """
        Process audio samples and return current speech state.
        
        Args:
            audio: Float32 PCM audio at 16kHz, normalized to [-1, 1]
            
        Returns:
            SpeechState indicating current detection status
        """
        # Accumulate audio in buffer
        self._audio_buffer = np.concatenate([self._audio_buffer, audio])
        
        # Process complete windows
        state = SpeechState.SILENCE if not self._is_speaking else SpeechState.SPEAKING
        
        while len(self._audio_buffer) >= self._window_size:
            window = self._audio_buffer[:self._window_size]
            self._audio_buffer = self._audio_buffer[self._window_size:]
            
            # Get speech probability from Silero VAD
            prob = self._get_speech_probability(window)
            
            # Update state based on probability
            state = self._update_state(prob)
        
        return state
    
    def _get_speech_probability(self, audio: np.ndarray) -> float:
        """Run Silero VAD on audio window and return speech probability."""
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Run VAD
        with torch.no_grad():
            prob = self.model(audio_tensor, self.config.sample_rate).item()
        
        return prob
    
    def _update_state(self, probability: float) -> SpeechState:
        """
        Update internal state based on speech probability.
        
        Implements hysteresis with patience thresholds to avoid
        rapid state transitions from momentary sounds or pauses.
        """
        current_time = time.time()
        is_speech = probability > self.config.threshold
        
        if not self._is_speaking:
            # Currently in silence, check for speech start
            if is_speech:
                if self._speech_start_time is None:
                    self._speech_start_time = current_time
                
                # Check if speech has lasted long enough
                speech_duration = current_time - self._speech_start_time
                if speech_duration >= self.config.start_patience:
                    self._is_speaking = True
                    self._speech_start_time = None
                    self._silence_start_time = None
                    return SpeechState.SPEECH_START
            else:
                # Reset speech start timer
                self._speech_start_time = None
            
            return SpeechState.SILENCE
        
        else:
            # Currently speaking, check for speech end
            if not is_speech:
                if self._silence_start_time is None:
                    self._silence_start_time = current_time
                
                # Check if silence has lasted long enough
                silence_duration = current_time - self._silence_start_time
                if silence_duration >= self.config.end_patience:
                    self._is_speaking = False
                    self._silence_start_time = None
                    self._speech_start_time = None
                    return SpeechState.SPEECH_END
            else:
                # Reset silence timer
                self._silence_start_time = None
            
            return SpeechState.SPEAKING
    
    @property
    def is_speaking(self) -> bool:
        """Whether the user is currently speaking."""
        return self._is_speaking
