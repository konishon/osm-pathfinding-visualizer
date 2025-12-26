"""
Sound effects for the search & path visualization.
Generates pinball/game room style beep sounds.
"""

import numpy as np
import pygame
from io import BytesIO


def generate_beep_sound(frequency: float, duration: float, sample_rate: int = 22050) -> pygame.mixer.Sound:
    """
    Generate a beep sound with the given frequency and duration.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz (default 22050)
    
    Returns:
        pygame.mixer.Sound object
    """
    # Generate the waveform
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = np.sin(2 * np.pi * frequency * t) * 0.3  # Reduce volume
    
    # Convert to 16-bit audio
    audio_data = (wave * 32767).astype(np.int16)
    
    # Create stereo (2 channels)
    stereo_data = np.zeros((len(audio_data), 2), dtype=np.int16)
    stereo_data[:, 0] = audio_data
    stereo_data[:, 1] = audio_data
    
    # Create pygame sound
    sound = pygame.mixer.Sound(buffer=stereo_data.tobytes())
    return sound


def create_search_sounds():
    """Create a library of sounds for the search phase (growing blue web)."""
    sounds = []
    
    # Pinball bumper sounds - ascending pitch
    frequencies = [400, 500, 600, 700]  # Hz
    
    for freq in frequencies:
        sound = generate_beep_sound(freq, 0.1)  # 100ms beeps
        sounds.append(sound)
    
    return sounds


def create_path_found_sound():
    """Create a triumphant sound for when the path is found."""
    pygame.mixer.init()
    
    # Create a rising tone followed by a happy ding
    sample_rate = 22050
    
    # Part 1: Rising tone (sweeping frequency)
    t1 = np.linspace(0, 0.3, int(sample_rate * 0.3))
    freq1 = np.linspace(600, 1200, len(t1))
    wave1 = np.sin(2 * np.pi * freq1 * t1) * 0.3
    
    # Part 2: Happy "ding" sound
    t2 = np.linspace(0, 0.4, int(sample_rate * 0.4))
    wave2 = (np.sin(2 * np.pi * 800 * t2) + 0.5 * np.sin(2 * np.pi * 1600 * t2)) * 0.25
    # Add decay
    decay = np.exp(-t2 * 3)
    wave2 = wave2 * decay
    
    # Combine
    combined = np.concatenate([wave1, wave2])
    
    # Convert to 16-bit audio
    audio_data = (combined * 32767).astype(np.int16)
    
    # Create stereo
    stereo_data = np.zeros((len(audio_data), 2), dtype=np.int16)
    stereo_data[:, 0] = audio_data
    stereo_data[:, 1] = audio_data
    
    sound = pygame.mixer.Sound(buffer=stereo_data.tobytes())
    return sound


def init_sound_system():
    """Initialize pygame mixer for sound playback."""
    try:
        pygame.mixer.init()
        return True
    except Exception as e:
        print(f"Warning: Could not initialize sound system: {e}")
        return False
