"""
Audio codec utilities for Twilio Media Streams.

Handles:
- mulaw (8-bit, 8kHz) encode/decode
- Resampling between 8kHz (Twilio) and 16kHz (Silero VAD)
- Loading and preparing response audio
"""

import base64
import struct
from pathlib import Path

import numpy as np

# audioop-lts provides mulaw encoding for Python 3.13+
import audioop


# Twilio sends/receives mulaw at 8kHz
TWILIO_SAMPLE_RATE = 8000
# Silero VAD expects 16kHz
VAD_SAMPLE_RATE = 16000


def decode_mulaw(data: bytes) -> np.ndarray:
    """
    Decode mulaw bytes to PCM float32 array.
    
    Args:
        data: Raw mulaw bytes (8-bit, 8kHz)
        
    Returns:
        PCM audio as float32 numpy array, normalized to [-1, 1]
    """
    # Convert mulaw to 16-bit PCM
    pcm_bytes = audioop.ulaw2lin(data, 2)
    
    # Convert to numpy array
    pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    
    # Normalize to float32 [-1, 1]
    return pcm_int16.astype(np.float32) / 32768.0


def encode_mulaw(pcm: np.ndarray) -> bytes:
    """
    Encode PCM float32 array to mulaw bytes.
    
    Args:
        pcm: PCM audio as float32 numpy array, normalized to [-1, 1]
        
    Returns:
        mulaw encoded bytes (8-bit, 8kHz)
    """
    # Convert float32 to int16
    pcm_int16 = (pcm * 32767).astype(np.int16)
    
    # Convert to bytes
    pcm_bytes = pcm_int16.tobytes()
    
    # Encode as mulaw
    return audioop.lin2ulaw(pcm_bytes, 2)


def resample(pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """
    Resample PCM audio using linear interpolation.
    
    Args:
        pcm: Input PCM audio as float32 numpy array
        from_rate: Source sample rate
        to_rate: Target sample rate
        
    Returns:
        Resampled audio as float32 numpy array
    """
    if from_rate == to_rate:
        return pcm
    
    # Calculate new length
    duration = len(pcm) / from_rate
    new_length = int(duration * to_rate)
    
    # Linear interpolation
    old_indices = np.arange(len(pcm))
    new_indices = np.linspace(0, len(pcm) - 1, new_length)
    
    return np.interp(new_indices, old_indices, pcm).astype(np.float32)


def upsample_for_vad(pcm: np.ndarray) -> np.ndarray:
    """Resample from Twilio 8kHz to VAD 16kHz."""
    return resample(pcm, TWILIO_SAMPLE_RATE, VAD_SAMPLE_RATE)


def downsample_for_twilio(pcm: np.ndarray) -> np.ndarray:
    """Resample from 16kHz to Twilio 8kHz."""
    return resample(pcm, VAD_SAMPLE_RATE, TWILIO_SAMPLE_RATE)


def load_response_audio(path) -> list:
    """
    Load a WAV file and convert to mulaw chunks for Twilio streaming.
    
    The audio is chunked into ~20ms segments (160 samples at 8kHz)
    for smooth real-time streaming.
    
    Args:
        path: Path to WAV file (any sample rate, will be resampled)
        
    Returns:
        List of base64-encoded mulaw chunks ready for Twilio
    """
    import wave
    
    path = Path(path)
    
    with wave.open(str(path), 'rb') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()
        
        # Read all frames
        raw_data = wav.readframes(n_frames)
    
    # Convert to numpy array based on sample width
    if sample_width == 1:
        pcm = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    elif sample_width == 2:
        pcm = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    
    # Convert stereo to mono by averaging channels
    if n_channels == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1)
    
    # Resample to 8kHz if needed
    if sample_rate != TWILIO_SAMPLE_RATE:
        pcm = resample(pcm, sample_rate, TWILIO_SAMPLE_RATE)
    
    # Chunk into ~20ms segments (160 samples at 8kHz)
    chunk_size = 160
    chunks = []
    
    for i in range(0, len(pcm), chunk_size):
        chunk_pcm = pcm[i:i + chunk_size]
        
        # Pad last chunk if needed
        if len(chunk_pcm) < chunk_size:
            chunk_pcm = np.pad(chunk_pcm, (0, chunk_size - len(chunk_pcm)))
        
        # Encode to mulaw and base64
        mulaw_bytes = encode_mulaw(chunk_pcm)
        b64_chunk = base64.b64encode(mulaw_bytes).decode('ascii')
        chunks.append(b64_chunk)
    
    return chunks
