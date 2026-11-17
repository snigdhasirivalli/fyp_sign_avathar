#!/usr/bin/env python
# coding: utf-8
"""
Generalized Hand Location Module for HamNoSys.

Fixes for cross-video consistency:
1. No horizontal frame flip: the old module flipped every frame, which swapped
   the anatomical side for file-based videos and broke other modules' handedness.
2. Works in a 3D palm-local frame (long axis, thumb-pinky axis, palm normal)
   instead of 2D screen coordinates, so hand rotation no longer changes labels.
3. Uses the other hand's closest fingertip as the contact point when two hands
   are present, and a self-contact fingertip otherwise.
4. The contacting side (palm / back / thumb side / pinky side) comes from the
   direction to the interacting hand in the palm-local frame, not from a
   contact point that is always inside the palm plane.
5. All distances are divided by palm scale (wrist -> middle MCP).
6. Final label is the mode of temporally smoothed frames; the old "one
   thumb-ball frame wins the whole video" rule amplified single-frame noise.

Output:
    {"per_frame": [...], "final": <label> | None}
"""

import os
import sys

import cv2
import numpy as np
from collections import Counter

try:
    import mediapipe as mp
except ImportError:
    mp = None

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
try:
    import mediapipe_solutions_compat  # noqa: F401
except ImportError:
    pass


HAND_TIPS = (4, 8, 12, 16, 20)


def _xyz(lm, w, h):
    return np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float32)


def _hand_points(hand_lm, w, h):
    return np.array([_xyz(lm, w, h) for lm in hand_lm.landmark], dtype=np.float32)


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros_like(v)


def _palm_scale(pts):
    return max(_dist(pts[0], pts[9]), 1e-4)


def _anatomical_side(palm_center_x, frame_width):
    # A signer facing the camera has their right hand on the image-left side.
    # This is stabler than MediaPipe handedness, which assumes a mirrored
    # selfie frame and flips on third-person recordings.
    return "R" if palm_center_x < frame_width * 0.5 else "L"


def _contact_point(hand_pts, other_pts):
    """Closest fingertip of the other hand, or a self-contact fingertip."""
    palm_center = hand_pts[[0, 5, 9, 13, 17]].mean(axis=0)
    scale = _palm_scale(hand_pts)

    if other_pts is not None:
        tips = [other_pts[i] for i in HAND_TIPS]
        tip = min(tips, key=lambda p: _dist(p, palm_center))
        # A hand resting elsewhere in the frame is not interacting with this one.
        if _dist(tip, palm_center) < 1.8 * scale:
            return tip

    tip = min((hand_pts[i] for i in HAND_TIPS), key=lambda p: _dist(p, palm_center))
    if _dist(tip, palm_center) > 0.95 * scale:
        return None
    return tip


def _other_palm(hand_pts, other_pts):
    """Palm centre of the interacting hand, or None if it is not close."""
    if other_pts is None:
        return None
    palm_center = hand_pts[[0, 5, 9, 13, 17]].mean(axis=0)
    other_center = other_pts[[0, 5, 9, 13, 17]].mean(axis=0)
    if _dist(other_center, palm_center) > 1.8 * _palm_scale(hand_pts):
        return None
    return other_center


def classify_contact_point(hand_pts, contact, side, other_center=None):
    wrist = hand_pts[0]
    index_mcp = hand_pts[5]
    middle_mcp = hand_pts[9]
    pinky_mcp = hand_pts[17]
    thumb_mcp = hand_pts[2]

    scale = _palm_scale(hand_pts)
    palm_center = hand_pts[[0, 5, 9, 13, 17]].mean(axis=0)

    # Palm-local axes
    u = _unit(index_mcp - pinky_mcp)          # positive towards the thumb side
    v = _unit(middle_mcp - wrist)             # positive towards the fingers
    normal = _unit(np.cross(u, v))            # positive out of the palm
    if side == "R":
        normal = -normal

    thumb_ball = 0.6 * thumb_mcp + 0.4 * wrist
    if _dist(contact, thumb_ball) / scale < 0.45:
        return "hamthumbball"

    # Which side of this hand faces the point of contact. This is only
    # meaningful when the other hand interacts; for a lone hand (fingertip
    # folded towards the palm) the palm is the contacting surface.
    if other_center is None:
        return "hampalm"

    direction = _unit(other_center - palm_center)

    normal_offset = float(np.dot(direction, normal))
    lateral = float(np.dot(direction, u))

    if normal_offset > 0.45:
        return "hampalm"
    if normal_offset < -0.45:
        return "hamhandback"
    if lateral > 0.45:
        return "hamthumbside"
    if lateral < -0.45:
        return "hampinkyside"
    return "hampalm"


def _classify_frame(hand_res, w, h):
    if hand_res is None or not hand_res.multi_hand_landmarks:
        return None

    hands = []
    for hand_lm in hand_res.multi_hand_landmarks:
        pts = _hand_points(hand_lm, w, h)
        side = _anatomical_side(float(pts[[0, 5, 9, 13, 17]].mean(axis=0)[0]), w)
        hands.append((pts, side))

    # The signing hand is the raised one; a resting hand must not become the
    # dominant hand merely because it is closer to the camera.
    hands.sort(key=lambda x: (x[0][0][1], -_palm_scale(x[0])))
    main_pts, side = hands[0]
    other_pts = hands[1][0] if len(hands) > 1 else None

    contact = _contact_point(main_pts, other_pts)
    if contact is None:
        return None
    return classify_contact_point(
        main_pts, contact, side, _other_palm(main_pts, other_pts))


def _smooth(values, window=5):
    smoothed = []
    n = len(values)
    for i in range(n):
        s = max(0, i - window // 2)
        e = min(n, i + window // 2 + 1)
        valid = [v for v in values[s:e] if v]
        smoothed.append(Counter(valid).most_common(1)[0][0] if valid else values[i])
    return smoothed


def run_hand_location_module(video_path, min_detection_confidence=0.4,
                             min_tracking_confidence=0.4):
    """
    Generalized hand location module.
    Returns:
        dict: {"per_frame": [...], "final": <label> | None}
    """
    if not os.path.exists(video_path):
        print(f"[Hand Location] Video not found: {video_path}")
        return {"per_frame": [], "final": None}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Hand Location] Failed to open video: {video_path}")
        return {"per_frame": [], "final": None}

    hands_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    raw = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_res = hands_detector.process(rgb)
        try:
            raw.append(_classify_frame(hand_res, w, h))
        except Exception:
            raw.append(None)

    cap.release()
    hands_detector.close()

    per_frame = _smooth(raw)
    valid = [v for v in per_frame if v]
    final = Counter(valid).most_common(1)[0][0] if valid else None
    return {"per_frame": per_frame, "final": final}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run generalized hand location module.")
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()
    print(run_hand_location_module(args.video_path))
