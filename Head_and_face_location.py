#!/usr/bin/env python
# coding: utf-8
"""
Generalized Head & Face Location Module for HamNoSys.

Fixes for cross-video consistency:
1. Regions are defined in face-box-relative coordinates with a halo wide enough
   for a hand held beside the head and long enough for under-chin/neck, so
   the result no longer depends on framing or zoom.
2. Chin, under-chin and neck regions are reachable again. The old function
   returned "hamcheek" for every rel_y >= 0.75, so those labels never occurred.
3. Mouth sub-labels use normalized motion statistics:
   stable vertical position -> teeth, upper mouth -> lips, large motion -> tongue.
4. Only the fingertip actually closest to the face is used, from either hand,
   instead of blindly taking the first hand and the index fingertip.
5. Per-video history and one aligned label per frame (no module-level deque).

Output:
    {"per_frame": [...], "final": <label> | None}
"""

import os
import sys

import cv2
import numpy as np
from collections import Counter, deque

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


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _face_bbox(face_landmarks, w, h):
    pts = np.array([_xyz(lm, w, h) for lm in face_landmarks.landmark], dtype=np.float32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return float(x1), float(y1), float(x2), float(y2)


def _nearest_fingertip(hand_res, w, h, face_cx, face_cy):
    """Return the fingertip closest to the face in face-relative units."""
    best = None
    if hand_res is None or not hand_res.multi_hand_landmarks:
        return None
    for hand_lm in hand_res.multi_hand_landmarks:
        for tip_idx in HAND_TIPS:
            tip = _xyz(hand_lm.landmark[tip_idx], w, h)
            lateral = np.hypot(tip[0] - face_cx, tip[1] - face_cy)
            if best is None or lateral < best[0]:
                best = (lateral, tip)
    return best


def _classify_region(bbox, tip, fy_history):
    x1, y1, x2, y2 = bbox[:4]
    face_w = max(x2 - x1, 1e-3)
    face_h = max(y2 - y1, 1e-3)

    fx, fy = tip[0], tip[1]
    rel_y = (fy - y1) / face_h
    rel_x = (fx - x1) / face_w

    # Allow a halo around the face wide enough for a hand held beside the head
    # and long enough for the under-chin / neck zone, but not the whole frame.
    in_x = -0.45 <= rel_x <= 1.45
    in_y = -0.15 <= rel_y <= 1.40
    if not (in_x and in_y):
        return None

    if rel_y < 0.05:
        return "hamheadtop"
    if rel_y < 0.22:
        return "hamforehead"

    if rel_y < 0.38:
        if rel_x < 0.12 or rel_x > 0.88:
            return "hamear"
        return "hameyes"

    if rel_y < 0.46:
        return "hamnose"
    if rel_y < 0.53:
        return "hamnostrils"

    if rel_y < 0.78:
        if not (0.22 <= rel_x <= 0.78):
            return "hamcheek"
        fy_history.append(fy)
        if len(fy_history) >= 5:
            y_range = max(fy_history) - min(fy_history)
        else:
            y_range = 0.0
        if y_range < 0.03 * face_h:
            return "hamteeth"
        mouth_y = y1 + 0.64 * face_h
        if fy < mouth_y - 0.03 * face_h:
            return "hamlips"
        return "hamtongue"

    if rel_y < 0.86:
        return "hamchin"
    if rel_y < 1.10:
        return "hamunderchin"
    return "hamneck"


def _smooth(values, window=5):
    smoothed = []
    n = len(values)
    for i in range(n):
        s = max(0, i - window // 2)
        e = min(n, i + window // 2 + 1)
        valid = [v for v in values[s:e] if v]
        smoothed.append(Counter(valid).most_common(1)[0][0] if valid else values[i])
    return smoothed


def run_head_face_location_module(video_path, min_detection_confidence=0.4,
                                  min_tracking_confidence=0.4):
    """
    Generalized head/face location module.
    Returns:
        dict: {"per_frame": [...], "final": <label> | None}
    """
    if not os.path.exists(video_path):
        print(f"[Head/Face] Video not found: {video_path}")
        return {"per_frame": [], "final": None}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Head/Face] Failed to open video: {video_path}")
        return {"per_frame": [], "final": None}

    face_detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    hands_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    fy_history = deque(maxlen=7)
    raw = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_res = face_detector.process(rgb)
        hand_res = hands_detector.process(rgb)

        label = None
        if face_res.multi_face_landmarks:
            bbox = _face_bbox(face_res.multi_face_landmarks[0], w, h)
            face_cx = (bbox[0] + bbox[2]) * 0.5
            face_cy = (bbox[1] + bbox[3]) * 0.5

            nearest = _nearest_fingertip(
                hand_res, w, h, face_cx, face_cy
            )
            if nearest is not None:
                _, tip = nearest
                label = _classify_region(bbox, tip, fy_history)
        raw.append(label)

    cap.release()
    face_detector.close()
    hands_detector.close()

    per_frame = _smooth(raw)
    valid = [v for v in per_frame if v]
    final = Counter(valid).most_common(1)[0][0] if valid else None
    return {"per_frame": per_frame, "final": final}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run generalized head/face location module.")
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()
    print(run_head_face_location_module(args.video_path))
