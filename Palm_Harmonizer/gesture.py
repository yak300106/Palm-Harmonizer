"""
gesture.py — Hand detection and open-palm recognition using MediaPipe Hands.

We use the landmark-based Hands solution (available in all mediapipe versions)
rather than the GestureRecognizer task (which requires a separate model file
download).  An "open palm" is detected geometrically: all five finger tips
must be extended away from the wrist.

Design goals:
  - Thread-safe: detector runs in the video thread only; result is shared via
    a threading.Lock-protected attribute on the detector object.
  - Confidence hysteresis: require N consecutive open-palm frames to activate,
    and N consecutive non-palm frames to deactivate.  This prevents flicker.
"""

import threading
import numpy as np
import cv2
import mediapipe as mp

# ---------------------------------------------------------------------------
# Tuneable parameters (edit here)
# ---------------------------------------------------------------------------
ACTIVATE_FRAMES   = 3    # consecutive open-palm frames needed to go ACTIVE
DEACTIVATE_FRAMES = 5    # consecutive non-palm frames needed to go INACTIVE
MIN_DETECTION_CONF = 0.7
MIN_TRACKING_CONF  = 0.6
# How far (as fraction of palm size) a fingertip must be above its MCP knuckle
EXTENSION_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Finger landmark indices (MediaPipe convention)
# ---------------------------------------------------------------------------
#  Each finger: [TIP, DIP, PIP, MCP]
FINGER_TIPS  = [4, 8, 12, 16, 20]   # thumb tip + 4 fingertips
FINGER_MCPS  = [2, 5,  9, 13, 17]   # corresponding MCP (knuckle) joints
WRIST_IDX    = 0


class GestureDetector:
    """
    Wraps MediaPipe Hands and exposes:
        .is_open_palm  → bool   (thread-safe read)
        .landmarks     → list of (x_px, y_px) or None
        .process(frame) → annotated frame
    """

    def __init__(self) -> None:
        self._lock           = threading.Lock()
        self._is_open_palm   = False
        self._landmarks_px: list[tuple[int, int]] | None = None

        # Hysteresis counters
        self._active_streak   = 0
        self._inactive_streak = 0
        self._currently_active = False

        # MediaPipe Hands
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_style = mp.solutions.drawing_styles
        self._hands    = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=MIN_DETECTION_CONF,
            min_tracking_confidence=MIN_TRACKING_CONF,
        )

    # ------------------------------------------------------------------
    # Public properties (thread-safe)
    # ------------------------------------------------------------------

    @property
    def is_open_palm(self) -> bool:
        with self._lock:
            return self._is_open_palm

    @property
    def landmarks_px(self) -> list[tuple[int, int]] | None:
        with self._lock:
            return self._landmarks_px

    # ------------------------------------------------------------------
    # Main processing method  (call from video thread)
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect hands in `frame`, update internal state, return annotated frame.
        Frame should be BGR uint8.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        detected_open = False
        lm_pixels: list[tuple[int, int]] | None = None

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]   # first hand only

            # Draw skeleton
            self._mp_draw.draw_landmarks(
                frame,
                hand_lms,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_style.get_default_hand_landmarks_style(),
                self._mp_style.get_default_hand_connections_style(),
            )

            # Convert to pixel coords
            lm_pixels = [
                (int(lm.x * w), int(lm.y * h))
                for lm in hand_lms.landmark
            ]

            detected_open = self._check_open_palm(hand_lms, w, h)

        # ---- Hysteresis ----
        if detected_open:
            self._active_streak   += 1
            self._inactive_streak  = 0
        else:
            self._inactive_streak += 1
            self._active_streak    = 0

        if not self._currently_active and self._active_streak >= ACTIVATE_FRAMES:
            self._currently_active = True
        elif self._currently_active and self._inactive_streak >= DEACTIVATE_FRAMES:
            self._currently_active = False

        with self._lock:
            self._is_open_palm  = self._currently_active
            self._landmarks_px  = lm_pixels

        return frame

    # ------------------------------------------------------------------
    # Internal geometry helpers
    # ------------------------------------------------------------------

    def _check_open_palm(self, hand_lms, w: int, h: int) -> bool:
        """
        Return True if all fingers appear extended (open palm).

        Strategy:
          For each finger, compare the TIP y-coordinate to the MCP y-coordinate
          (in image space, smaller y = higher on screen).  We also normalise by
          the palm size (wrist-to-middle-MCP distance) so it works at any scale.

          Special case: thumb uses x-axis distance instead of y-axis.
        """
        lm = hand_lms.landmark

        # Palm size = distance from wrist (0) to middle-finger MCP (9)
        wx, wy = lm[WRIST_IDX].x * w, lm[WRIST_IDX].y * h
        mx, my = lm[9].x * w,         lm[9].y * h
        palm_size = np.hypot(mx - wx, my - wy) + 1e-6

        extended_count = 0

        for i, (tip_idx, mcp_idx) in enumerate(zip(FINGER_TIPS, FINGER_MCPS)):
            tip = lm[tip_idx]
            mcp = lm[mcp_idx]

            if i == 0:  # Thumb: check x-distance from wrist
                dist = abs(tip.x * w - wx) / palm_size
            else:        # Other fingers: tip should be above MCP in y
                dist = (mcp.y * h - tip.y * h) / palm_size  # positive = extended

            if dist > EXTENSION_THRESHOLD:
                extended_count += 1

        # Require at least 4 of 5 fingers extended
        return extended_count >= 4

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
