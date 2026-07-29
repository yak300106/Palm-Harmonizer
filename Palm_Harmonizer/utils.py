"""
utils.py — Utility functions for Palm Harmonizer
  - Hz → MIDI note number
  - Hz → note name (e.g. "A4", "C#3")
  - OpenCV drawing helpers (text with shadow, semi-transparent overlays)
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Music / pitch helpers
# ---------------------------------------------------------------------------

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

A4_HZ   = 440.0          # reference pitch
A4_MIDI = 69             # MIDI note number for A4


def hz_to_midi(freq_hz: float) -> float:
    """Convert frequency in Hz to a (possibly fractional) MIDI note number."""
    if freq_hz <= 0:
        return 0.0
    return 12.0 * np.log2(freq_hz / A4_HZ) + A4_MIDI


def midi_to_hz(midi_note: float) -> float:
    """Convert a MIDI note number (can be fractional) to Hz."""
    return A4_HZ * 2.0 ** ((midi_note - A4_MIDI) / 12.0)


def hz_to_note_name(freq_hz: float) -> str:
    """Return a human-readable note name like 'A4' or 'C#3' for a given Hz."""
    if freq_hz <= 0:
        return "---"
    midi = round(hz_to_midi(freq_hz))
    octave = (midi // 12) - 1
    name   = NOTE_NAMES[midi % 12]
    return f"{name}{octave}"


def semitones_to_ratio(semitones: float) -> float:
    """Return the frequency ratio corresponding to N semitones."""
    return 2.0 ** (semitones / 12.0)


# Harmony interval definitions (semitones above root)
HARMONY_MODES = {
    "major":  [4, 7, 12],          # major 3rd, perfect 5th, octave
    "minor":  [3, 7, 12],          # minor 3rd, perfect 5th, octave
    "power":  [7, 12, 19],         # 5th, octave, 5th above octave
    "major7": [4, 7, 11, 12],      # major 7th chord
    "sus2":   [2, 7, 12],          # suspended 2nd
}


# ---------------------------------------------------------------------------
# OpenCV drawing helpers
# ---------------------------------------------------------------------------

def put_text_shadow(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    shadow_offset: int = 2,
) -> None:
    """Draw text with a dark drop-shadow for readability on any background."""
    x, y = pos
    font = cv2.FONT_HERSHEY_DUPLEX
    # Shadow
    cv2.putText(frame, text, (x + shadow_offset, y + shadow_offset),
                font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Main text
    cv2.putText(frame, text, (x, y),
                font, font_scale, color, thickness, cv2.LINE_AA)


def draw_semi_transparent_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.45,
) -> None:
    """Draw a filled rectangle blended onto the frame (alpha compositing)."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_border(
    frame: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 6,
) -> None:
    """Draw a colored border around the entire frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)


def draw_pill(
    frame: np.ndarray,
    text: str,
    cx: int, cy: int,
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.6,
    padding_x: int = 18,
    padding_y: int = 10,
    alpha: float = 0.75,
) -> None:
    """Draw a rounded-rect 'pill' label centered at (cx, cy)."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
    x1 = cx - tw // 2 - padding_x
    y1 = cy - th // 2 - padding_y
    x2 = cx + tw // 2 + padding_x
    y2 = cy + th // 2 + padding_y
    draw_semi_transparent_rect(frame, x1, y1, x2, y2, bg_color, alpha)
    # Rounded corners illusion via ellipses
    r = min(padding_y, 12)
    for ex, ey in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.ellipse(frame, (ex, ey), (r, r), 0, 0, 360, bg_color, -1)
    put_text_shadow(frame, text, (cx - tw // 2, cy + th // 2),
                    font_scale, text_color, 1)


def draw_waveform_bar(
    frame: np.ndarray,
    energy: float,          # 0.0 – 1.0
    x: int, y: int,
    width: int = 120,
    height: int = 12,
    color_low: tuple = (0, 200, 100),
    color_high: tuple = (0, 80, 255),
) -> None:
    """Draw a simple energy meter bar (green → blue)."""
    energy = float(np.clip(energy, 0.0, 1.0))
    t = energy
    color = tuple(int(color_low[i] * (1 - t) + color_high[i] * t) for i in range(3))
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + int(width * energy), y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)
