"""
main.py — Palm Harmonizer: orchestration, video loop, and UI overlay.

Run:
    python main.py

Keyboard controls (while the OpenCV window is focused):
    Q / ESC   → quit
    R         → reset (recalibrate gesture hysteresis counters)
    H         → toggle hearing your own voice through speakers
    V         → toggle Voice Activity Detection (VAD)
    M         → cycle harmony mode  (major → minor → power → major7 → sus2)
    S         → save a screenshot of the current frame

See README.md for full setup instructions.
"""

import sys
import time
import threading
import cv2
import numpy as np

from gesture        import GestureDetector
from audio_processor import AudioProcessor
from utils import (
    hz_to_note_name, midi_to_hz, hz_to_midi,
    put_text_shadow, draw_semi_transparent_rect, draw_border,
    draw_pill, draw_waveform_bar, HARMONY_MODES,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_INDEX   = 0
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
WINDOW_NAME    = "🖐  Palm Harmonizer"

HARMONY_MODE_CYCLE = list(HARMONY_MODES.keys())

# UI colours (BGR)
COLOR_ACTIVE   = (0,  230, 100)   # vivid green
COLOR_INACTIVE = (50, 50,  50)    # dark grey
COLOR_ACCENT   = (255, 140, 0)    # orange
COLOR_NOTE     = (255, 200, 50)   # warm yellow
COLOR_TEXT     = (240, 240, 240)
COLOR_DIM      = (120, 120, 120)
COLOR_BORDER_ON  = (0, 220, 90)
COLOR_BORDER_OFF = (30,  30, 30)


# ---------------------------------------------------------------------------
# FPS counter helper
# ---------------------------------------------------------------------------
class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._times: list[float] = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ---------------------------------------------------------------------------
# UI drawing
# ---------------------------------------------------------------------------

def draw_ui(
    frame:          np.ndarray,
    is_harmonizing: bool,
    pitch_hz:       float | None,
    active_notes:   list[float],
    harmony_mode:   str,
    energy:         float,
    hear_self:      bool,
    vad_enabled:    bool,
    fps:            float,
) -> None:
    """Render all HUD elements onto `frame` in-place."""
    h, w = frame.shape[:2]

    # === Coloured border indicates harmonizing state ===
    border_color = COLOR_BORDER_ON if is_harmonizing else COLOR_BORDER_OFF
    draw_border(frame, border_color, thickness=8 if is_harmonizing else 3)

    # === Top-left: status pill ===
    if is_harmonizing:
        status_text = "  HARMONIZING  "
        pill_color  = (0, 160, 60)
    else:
        status_text = "  WAITING FOR PALM  "
        pill_color  = (30, 30, 30)
    draw_pill(frame, status_text, w // 2, 38, pill_color,
              font_scale=0.75, padding_x=24, padding_y=12, alpha=0.80)

    # === Top-right: FPS ===
    put_text_shadow(frame, f"FPS {fps:.0f}", (w - 110, 32),
                    font_scale=0.55, color=COLOR_DIM, thickness=1)

    # === Left panel: pitch & notes info ===
    panel_x = 18
    draw_semi_transparent_rect(frame, panel_x, 70, panel_x + 220, 300,
                               (0, 0, 0), alpha=0.50)

    # Root note
    if pitch_hz and pitch_hz > 0:
        root_name = hz_to_note_name(pitch_hz)
        put_text_shadow(frame, "ROOT", (panel_x + 10, 100),
                        0.50, COLOR_DIM, 1)
        put_text_shadow(frame, root_name, (panel_x + 10, 140),
                        1.2, COLOR_NOTE, 2)
        put_text_shadow(frame, f"{pitch_hz:.1f} Hz", (panel_x + 10, 165),
                        0.50, COLOR_DIM, 1)
    else:
        put_text_shadow(frame, "ROOT", (panel_x + 10, 100), 0.50, COLOR_DIM, 1)
        put_text_shadow(frame, "---", (panel_x + 10, 140), 1.2, COLOR_DIM, 1)

    # Harmony notes
    put_text_shadow(frame, "HARMONIES", (panel_x + 10, 195), 0.50, COLOR_DIM, 1)
    if active_notes:
        intervals = HARMONY_MODES.get(harmony_mode, [])
        interval_names = ["M3", "5th", "Oct", "M7", "2nd"]
        for i, freq in enumerate(active_notes[:4]):
            lbl = interval_names[i] if i < len(interval_names) else f"+{intervals[i] if i < len(intervals) else '?'}st"
            note = hz_to_note_name(freq)
            put_text_shadow(frame, f"{lbl}  {note}", (panel_x + 10, 220 + i * 24),
                            0.58, COLOR_ACTIVE, 1)
    else:
        put_text_shadow(frame, "—", (panel_x + 10, 220), 0.6, COLOR_DIM, 1)

    # === Bottom-left: energy bar + mode + toggles ===
    bar_y = h - 95
    put_text_shadow(frame, "VOICE", (panel_x, bar_y - 14), 0.45, COLOR_DIM, 1)
    draw_waveform_bar(frame, energy, panel_x, bar_y, width=160, height=10)

    put_text_shadow(frame, f"MODE: {harmony_mode.upper()}  [M to cycle]",
                    (panel_x, h - 65), 0.48, COLOR_ACCENT, 1)
    hear_str = "ON" if hear_self  else "off"
    vad_str  = "ON" if vad_enabled else "off"
    put_text_shadow(frame, f"HEAR SELF: {hear_str}  [H]    VAD: {vad_str}  [V]",
                    (panel_x, h - 42), 0.45, COLOR_DIM, 1)
    put_text_shadow(frame, "Q/ESC: quit   R: reset   S: screenshot",
                    (panel_x, h - 20), 0.42, COLOR_DIM, 1)

    # === Centre: big "wave" animation when harmonizing ===
    if is_harmonizing and pitch_hz and pitch_hz > 0:
        _draw_wave_animation(frame, energy, w // 2, h - 50, width=300)


def _draw_wave_animation(
    frame: np.ndarray,
    energy: float,
    cx: int, cy: int,
    width: int = 300,
) -> None:
    """Draw a simple animated sine wave in the bottom centre."""
    n_pts = 80
    amp   = int(20 * max(energy, 0.15))
    t     = time.perf_counter()
    pts   = []
    for i in range(n_pts):
        x = cx - width // 2 + int(i * width / n_pts)
        y = cy + int(amp * np.sin(2 * np.pi * (i / n_pts * 3 + t * 4)))
        pts.append((x, y))

    for i in range(len(pts) - 1):
        alpha_t = i / (len(pts) - 1)
        r = int(0 * (1 - alpha_t) + 0 * alpha_t)
        g = int(200 * (1 - alpha_t) + 120 * alpha_t)
        b = int(100 * (1 - alpha_t) + 255 * alpha_t)
        cv2.line(frame, pts[i], pts[i + 1], (b, g, r), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  🖐  Palm Harmonizer  —  starting up")
    print("=" * 60)

    # --- Camera ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in main.py.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"[OK] Camera opened ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")

    # --- Gesture detector ---
    gesture = GestureDetector()
    print("[OK] Gesture detector ready")

    # --- Audio processor ---
    audio = AudioProcessor()
    try:
        audio.start()
        print("[OK] Audio stream started")
    except Exception as e:
        print(f"[ERROR] Could not start audio: {e}")
        cap.release()
        sys.exit(1)

    fps_counter  = FPSCounter()
    mode_idx     = 0   # index into HARMONY_MODE_CYCLE
    screenshot_n = 0

    print("\nControls: Q/ESC=quit  R=reset  H=hear-self  V=VAD  M=mode  S=screenshot")
    print("Show an open palm to the camera to start harmonizing!\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed — retrying...")
                time.sleep(0.01)
                continue

            # Mirror the image so it feels like a mirror
            frame = cv2.flip(frame, 1)

            # --- Run gesture detection ---
            frame = gesture.process(frame)

            # --- Sync harmonizing state from gesture to audio ---
            palm_open = gesture.is_open_palm
            audio.set_harmonizing(palm_open)

            # --- Read audio state for UI ---
            pitch_hz     = audio.current_pitch_hz
            active_notes = audio.active_note_freqs
            energy       = audio.energy
            fps          = fps_counter.tick()

            # --- Draw HUD ---
            draw_ui(
                frame        = frame,
                is_harmonizing = palm_open,
                pitch_hz     = pitch_hz,
                active_notes = active_notes,
                harmony_mode = audio.harmony_mode,
                energy       = energy,
                hear_self    = audio.hear_self,
                vad_enabled  = audio.vad_enabled,
                fps          = fps,
            )

            cv2.imshow(WINDOW_NAME, frame)

            # --- Keyboard handling ---
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # Q or ESC
                break

            elif key == ord("r"):       # Reset
                gesture._active_streak   = 0
                gesture._inactive_streak = 0
                gesture._currently_active = False
                audio._smoothed_pitch    = 0.0
                print("[R] Reset — gesture + pitch state cleared")

            elif key == ord("h"):       # Toggle hear self
                state = audio.toggle_hear_self()
                print(f"[H] Hear self: {'ON' if state else 'off'}")

            elif key == ord("v"):       # Toggle VAD
                state = audio.toggle_vad()
                print(f"[V] VAD: {'ON' if state else 'off'}")

            elif key == ord("m"):       # Cycle harmony mode
                mode_idx = (mode_idx + 1) % len(HARMONY_MODE_CYCLE)
                new_mode = HARMONY_MODE_CYCLE[mode_idx]
                audio.set_harmony_mode(new_mode)
                print(f"[M] Harmony mode → {new_mode}")

            elif key == ord("s"):       # Screenshot
                fname = f"screenshot_{screenshot_n:03d}.png"
                cv2.imwrite(fname, frame)
                screenshot_n += 1
                print(f"[S] Saved {fname}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        print("[INFO] Shutting down...")
        audio.stop()
        gesture.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Goodbye! 🖐")


if __name__ == "__main__":
    main()
