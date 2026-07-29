"""
synthesizer.py — Real-time additive sine-wave synthesizer for harmony voices.

Design:
  - Each "voice" is a continuous sine oscillator that tracks a target frequency.
  - Frequencies are updated from the audio callback thread (fast path).
  - Amplitude envelopes (attack / release) prevent clicks on start/stop.
  - The synthesizer fills an output buffer of arbitrary length on each callback.

All state is kept in numpy arrays; no Python loops in the hot path.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Tuneable parameters
# ---------------------------------------------------------------------------
ATTACK_MS   = 25.0   # amplitude ramp-up time in ms
RELEASE_MS  = 60.0   # amplitude ramp-down time in ms
VOICE_GAIN  = 0.45 # per-voice amplitude (tune so sum of 4 voices ≈ 0.85)
MAX_VOICES  = 4      # maximum simultaneous harmony voices


class SineVoice:
    """
    A single continuously-running sine oscillator with amplitude envelope.

    The oscillator phase is maintained across buffer fills to avoid
    discontinuities at buffer boundaries.
    """

    def __init__(self, sample_rate: int) -> None:
        self.sr          = sample_rate
        self.freq        = 0.0          # current target frequency (Hz)
        self.phase       = 0.0          # current phase in radians
        self.amplitude   = 0.0          # current instantaneous amplitude
        self.target_amp  = 0.0          # 0.0 = silent, VOICE_GAIN = active
        self._attack_rate  = VOICE_GAIN / (ATTACK_MS  * 0.001 * sample_rate)
        self._release_rate = VOICE_GAIN / (RELEASE_MS * 0.001 * sample_rate)

    def activate(self, freq: float) -> None:
        """Start (or retarget) this voice to the given frequency."""
        self.freq       = freq
        self.target_amp = VOICE_GAIN

    def deactivate(self) -> None:
        """Begin fade-out of this voice."""
        self.target_amp = 0.0

    def is_silent(self) -> bool:
        return self.amplitude < 1e-6 and self.target_amp == 0.0

    def fill(self, n_samples: int) -> np.ndarray:
        """
        Generate `n_samples` of audio for this voice.
        Returns a float32 array.
        """
        out = np.zeros(n_samples, dtype=np.float32)
        if self.is_silent():
            return out

        # Build per-sample amplitude envelope
        amps = np.empty(n_samples, dtype=np.float32)
        amp  = self.amplitude
        for i in range(n_samples):
            if amp < self.target_amp:
                amp = min(amp + self._attack_rate, self.target_amp)
            elif amp > self.target_amp:
                amp = max(amp - self._release_rate, self.target_amp)
            amps[i] = amp
        self.amplitude = amp

        if self.freq <= 0:
            return out

        # Generate sine wave with accumulated phase
        t = np.arange(n_samples, dtype=np.float64)
        phase_inc = 2.0 * np.pi * self.freq / self.sr
        phases    = self.phase + phase_inc * t
        out[:] = (np.sin(phases) * amps).astype(np.float32)

        # Update persistent phase (keep in [0, 2π] to avoid float drift)
        self.phase = (phases[-1] + phase_inc) % (2.0 * np.pi)
        return out


class Synthesizer:
    """
    Manages multiple SineVoices and mixes them into a single mono buffer.

    Usage pattern (from audio callback):
        synth.set_harmonies([220.0, 277.18, 329.63])   # update target freqs
        # or
        synth.silence()                                  # fade everything out
        buffer = synth.render(n_frames)                  # get mixed audio
    """

    def __init__(self, sample_rate: int, n_voices: int = MAX_VOICES) -> None:
        self.sr     = sample_rate
        self.voices = [SineVoice(sample_rate) for _ in range(n_voices)]
        self._active_freqs: list[float] = []

    # ------------------------------------------------------------------
    # Control API  (called from audio callback — keep it fast)
    # ------------------------------------------------------------------

    def set_harmonies(self, freqs: list[float]) -> None:
        """
        Update voices to play the given list of frequencies.
        Extra voices beyond len(freqs) are deactivated.
        """
        freqs = freqs[: len(self.voices)]
        for i, voice in enumerate(self.voices):
            if i < len(freqs):
                voice.activate(freqs[i])
            else:
                voice.deactivate()
        self._active_freqs = list(freqs)

    def silence(self) -> None:
        """Fade out all voices."""
        for v in self.voices:
            v.deactivate()
        self._active_freqs = []

    # ------------------------------------------------------------------
    # Render  (called from audio callback)
    # ------------------------------------------------------------------

    def render(self, n_frames: int) -> np.ndarray:
        """
        Mix all voices and return a float32 mono array of length n_frames.
        Values are in [-1, 1] (soft-clipped).
        """
        mix = np.zeros(n_frames, dtype=np.float32)
        for v in self.voices:
            mix += v.fill(n_frames)
        # Soft clip via tanh to prevent hard clipping artefacts
        return np.tanh(mix).astype(np.float32)

    @property
    def active_freqs(self) -> list[float]:
        return list(self._active_freqs)
