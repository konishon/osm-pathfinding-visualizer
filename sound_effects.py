"""
Sound effects for the search & path visualization.
Generates pinball/game room style beep sounds.
"""

import numpy as np
import pygame
from io import BytesIO

def get_beep_waveform(frequency: float, duration: float, sample_rate: int = 22050) -> np.ndarray:
    """Generate raw waveform for a beep."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = np.sin(2 * np.pi * frequency * t) * 0.3  # Reduce volume
    return (wave * 32767).astype(np.int16)

def generate_beep_sound(frequency: float, duration: float, sample_rate: int = 22050) -> pygame.mixer.Sound:
    """
    Generate a beep sound with the given frequency and duration.
    """
    audio_data = get_beep_waveform(frequency, duration, sample_rate)
    
    # Create stereo (2 channels)
    stereo_data = np.zeros((len(audio_data), 2), dtype=np.int16)
    stereo_data[:, 0] = audio_data
    stereo_data[:, 1] = audio_data
    
    # Create pygame sound
    sound = pygame.mixer.Sound(buffer=stereo_data.tobytes())
    return sound


def create_search_sounds(num_steps: int = 4, start_freq: float = 200, end_freq: float = 1200):
    """Create a library of sounds for the search phase (growing blue web)."""
    sounds = []
    
    # Ascending pitch
    if num_steps <= 1:
        frequencies = [start_freq]
    else:
        frequencies = np.linspace(start_freq, end_freq, num_steps)
    
    for freq in frequencies:
        sound = generate_beep_sound(freq, 0.1)  # 100ms beeps
        sounds.append(sound)
    
    return sounds

def get_search_waveforms(num_steps: int = 4, start_freq: float = 200, end_freq: float = 1200):
    """Return raw waveforms for search sounds."""
    waveforms = []
    if num_steps <= 1:
        frequencies = [start_freq]
    else:
        frequencies = np.linspace(start_freq, end_freq, num_steps)
        
    for freq in frequencies:
        waveforms.append(get_beep_waveform(freq, 0.1))
    return waveforms

def get_path_found_waveform(sample_rate: int = 22050) -> np.ndarray:
    """Generate raw waveform for path found sound."""
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
    return (combined * 32767).astype(np.int16)

def create_path_found_sound():
    """Create a triumphant sound for when the path is found."""
    pygame.mixer.init()
    
    audio_data = get_path_found_waveform()
    
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
