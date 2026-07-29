"""
audio_processor.py — Microphone capture, pitch detection, and audio output.
Pitch detection uses a pure numpy/scipy YIN implementation — no aubio needed.
"""

import threading
import numpy as np
import sounddevice as sd
from typing import Optional

from synthesizer import Synthesizer
from utils import hz_to_midi, midi_to_hz, HARMONY_MODES

# ---------------------------------------------------------------------------
# Tuneable parameters
# ---------------------------------------------------------------------------
SAMPLE_RATE     = 44100
BLOCK_SIZE      = 512
PITCH_WIN_SIZE  = 2048
HOP_SIZE        = BLOCK_SIZE
VAD_RMS_THRESH  = 0.008
PITCH_CONF_MIN  = 0.65
PITCH_SMOOTH    = 0.25
PITCH_RANGE_HZ  = (60, 1200)


# ---------------------------------------------------------------------------
# Pure numpy/scipy YIN pitch detector (no aubio required)
# ---------------------------------------------------------------------------

def yin_pitch(buffer: np.ndarray, sample_rate: int,
              f_min: float = 60.0, f_max: float = 1200.0,
              threshold: float = 0.15) -> tuple[float, float]:
    """
    YIN pitch detection algorithm (de Cheveigné & Kawahara 2002).
    Returns (pitch_hz, confidence). pitch_hz=0 means no pitch found.
    """
    N = len(buffer)
    tau_min = int(sample_rate / f_max)
    tau_max = int(sample_rate / f_min)
    tau_max = min(tau_max, N // 2)

    if tau_min >= tau_max:
        return 0.0, 0.0

    # Difference function
    diff = np.zeros(tau_max)
    for tau in range(1, tau_max):
        diff[tau] = np.sum((buffer[:N - tau] - buffer[tau:N]) ** 2)

    # Cumulative mean normalised difference
    cmnd = np.zeros(tau_max)
    cmnd[0] = 1.0
    running_sum = 0.0
    for tau in range(1, tau_max):
        running_sum += diff[tau]
        if running_sum == 0:
            cmnd[tau] = 1.0
        else:
            cmnd[tau] = diff[tau] * tau / running_sum

    # Find first dip below threshold
    tau_est = -1
    for tau in range(tau_min, tau_max - 1):
        if cmnd[tau] < threshold:
            while tau + 1 < tau_max - 1 and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            tau_est = tau
            break

    if tau_est == -1:
        tau_est = tau_min + int(np.argmin(cmnd[tau_min:tau_max]))

    # Parabolic interpolation
    if 0 < tau_est < tau_max - 1:
        s0, s1, s2 = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
        denom = (s0 - 2 * s1 + s2)
        if denom != 0:
            tau_est = tau_est + (s0 - s2) / (2 * denom)

    pitch_hz   = sample_rate / tau_est if tau_est > 0 else 0.0
    confidence = float(np.clip(1.0 - float(np.interp(tau_est,
                                np.arange(tau_max), cmnd)), 0.0, 1.0))

    if not (f_min <= pitch_hz <= f_max):
        return 0.0, 0.0

    return pitch_hz, confidence


# ---------------------------------------------------------------------------
# AudioProcessor
# ---------------------------------------------------------------------------

class AudioProcessor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._harmonizing   = False
        self._current_pitch: Optional[float] = None
        self._active_notes:  list[float]     = []
        self._energy         = 0.0
        self._hear_self      = False
        self._vad_enabled    = True
        self._harmony_mode   = "major"
        self._synth          = Synthesizer(SAMPLE_RATE)
        self._pitch_buf      = np.zeros(PITCH_WIN_SIZE, dtype=np.float32)
        self._smoothed_pitch = 0.0
        self._stream: Optional[sd.Stream] = None

    def start(self) -> None:
        self._stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            channels=1,
            callback=self._audio_callback,
            latency="low",
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_harmonizing(self, active: bool) -> None:
        with self._lock:
            self._harmonizing = active
        if not active:
            self._synth.silence()

    def set_harmony_mode(self, mode: str) -> None:
        if mode in HARMONY_MODES:
            with self._lock:
                self._harmony_mode = mode

    def toggle_hear_self(self) -> bool:
        with self._lock:
            self._hear_self = not self._hear_self
            return self._hear_self

    def toggle_vad(self) -> bool:
        with self._lock:
            self._vad_enabled = not self._vad_enabled
            return self._vad_enabled

    @property
    def current_pitch_hz(self) -> Optional[float]:
        with self._lock:
            return self._current_pitch

    @property
    def active_note_freqs(self) -> list[float]:
        with self._lock:
            return list(self._active_notes)

    @property
    def energy(self) -> float:
        with self._lock:
            return self._energy

    @property
    def is_harmonizing(self) -> bool:
        with self._lock:
            return self._harmonizing

    @property
    def hear_self(self) -> bool:
        with self._lock:
            return self._hear_self

    @property
    def vad_enabled(self) -> bool:
        with self._lock:
            return self._vad_enabled

    @property
    def harmony_mode(self) -> str:
        with self._lock:
            return self._harmony_mode

    def _audio_callback(self, indata, outdata, frames, time, status) -> None:
        mic_mono = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(mic_mono ** 2)))
        norm_energy = min(rms / 0.1, 1.0)

        with self._lock:
            harmonizing  = self._harmonizing
            vad_enabled  = self._vad_enabled
            hear_self    = self._hear_self
            harmony_mode = self._harmony_mode
            self._energy = norm_energy

        pitch_hz: Optional[float] = None
        voice_active = (not vad_enabled) or (rms > VAD_RMS_THRESH)

        if harmonizing and voice_active:
            self._pitch_buf = np.roll(self._pitch_buf, -frames)
            self._pitch_buf[-frames:] = mic_mono

            raw_pitch, confidence = yin_pitch(
                self._pitch_buf, SAMPLE_RATE,
                f_min=PITCH_RANGE_HZ[0],
                f_max=PITCH_RANGE_HZ[1],
                threshold=0.15,
            )

            if confidence >= PITCH_CONF_MIN and raw_pitch > 0:
                if self._smoothed_pitch == 0.0:
                    self._smoothed_pitch = raw_pitch
                else:
                    self._smoothed_pitch = (
                        PITCH_SMOOTH * raw_pitch +
                        (1 - PITCH_SMOOTH) * self._smoothed_pitch
                    )
                pitch_hz = self._smoothed_pitch
            else:
                self._smoothed_pitch *= 0.95

        if harmonizing and pitch_hz is not None and voice_active:
            intervals  = HARMONY_MODES.get(harmony_mode, HARMONY_MODES["major"])
            root_midi  = hz_to_midi(pitch_hz)
            harm_freqs = [midi_to_hz(root_midi + s) for s in intervals]
            self._synth.set_harmonies(harm_freqs)
            active_notes = harm_freqs
        else:
            if not harmonizing:
                self._synth.silence()
                active_notes = []
            else:
                active_notes = self._synth.active_freqs

        harmony_audio = self._synth.render(frames)

        if hear_self:
            outdata[:, 0] = harmony_audio + mic_mono * 0.7
        else:
            outdata[:, 0] = harmony_audio

        np.clip(outdata, -0.95, 0.95, out=outdata)

        with self._lock:
            self._current_pitch = pitch_hz
            self._active_notes  = active_notes