#!/usr/bin/env python
# coding: utf-8
"""
Generalized Contact-Type Module for HamNoSys.

Fixes for cross-video consistency:
1. All thresholds are relative to palm scale (wrist -> middle MCP) instead of
   raw MediaPipe normalized units, which vary with resolution and camera distance.
2. Uses 3D fingertip sets and palm centres, not only the index fingertips.
3. A single hand no longer produces a fake "hamtouch" by comparing a point
   with itself.
4. Hand-to-body contact (chest/shoulder/face silhouettes) is detected, which
   covers the many one-handed signs in ISL that touch the body.
5. Cross/brushing use temporal history instead of a single frame, and depth is
   checked for behind/interlock.
6. Final label comes from smoothed valid contact frames; "no-contact" does not
   automatically win when contact exists.

Output:
    {"per_frame": [...], "final": <label>}
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
NO_CONTACT = "no-contact"


def _xyz(lm, w, h):
    return np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float32)


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _hand_points(hand_lm, w, h):
    return np.array([_xyz(lm, w, h) for lm in hand_lm.landmark], dtype=np.float32)


def _palm_scale(pts):
    return max(_dist(pts[0], pts[9]), 1e-4)


def _palm_center(pts):
    return pts[[0, 5, 9, 13, 17]].mean(axis=0)


def _tips(pts):
    return [pts[i] for i in HAND_TIPS]


def _pose_points(pose_landmarks, w, h):
    if pose_landmarks is None:
        return None
    return np.array([_xyz(lm, w, h) for lm in pose_landmarks.landmark], dtype=np.float32)


def _mean_motion(history):
    if len(history) < 2:
        return 0.0
    return float(np.mean([_dist(history[i], history[i + 1]) for i in range(len(history) - 1)]))


# ==========================================================
# TWO-HAND CONTACT
# ==========================================================

def _interlock(a, b, scale):
    close_pairs = 0
    depth_signs = []
    for ta in _tips(a):
        for tb in _tips(b):
            if _dist(ta, tb) < 0.45 * scale:
                close_pairs += 1
                depth_signs.append(1 if (ta[2] - tb[2]) > 0 else -1)
    if close_pairs < 2:
        return False
    return len(set(depth_signs)) > 1


def _two_hand_label(a, b, a_hist, b_hist, order_history):
    scale = 0.5 * (_palm_scale(a) + _palm_scale(b))
    pa, pb = _palm_center(a), _palm_center(b)

    min_tip = min(_dist(ta, tb) for ta in _tips(a) for tb in _tips(b))
    palm_d = _dist(pa, pb)
    depth_diff = abs(pa[2] - pb[2]) / scale

    if _interlock(a, b, scale):
        return "haminterlock"

    # Cross: the two palms swap horizontal order while separated in depth.
    if (len(order_history) >= 4 and len(set(order_history)) > 1
            and depth_diff > 0.25 and min_tip < 0.90 * scale):
        return "hamcross"

    # Brushing: touching while both hands keep moving.
    motion = 0.5 * (_mean_motion(a_hist) + _mean_motion(b_hist)) / scale
    if min_tip < 0.55 * scale and motion > 0.05:
        return "hambrushing"

    if min_tip < 0.28 * scale or palm_d < 0.45 * scale:
        return "hamtouch"

    if min_tip < 0.80 * scale or palm_d < 1.05 * scale:
        return "hamclose"

    if depth_diff > 0.65 and min_tip < 1.10 * scale:
        return "hambehind"

    return None


# ==========================================================
# HAND / BODY CONTACT
# ==========================================================

def _dist2(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _body_contact(palm, pose_pts):
    """Hand-to-body contact from body-plane (2D) geometry.

    MediaPipe's pose and hand models estimate z in different, non-comparable
    reference frames, so depth is deliberately not compared across models.
    """
    if pose_pts is None:
        return None

    ls, rs = pose_pts[11], pose_pts[12]
    lh, rh = pose_pts[23], pose_pts[24]
    nose = pose_pts[0]

    shoulder_mid = (ls + rs) * 0.5
    hip_mid = (lh + rh) * 0.5
    shoulder_width = max(_dist(ls, rs), 1e-3)
    torso_height = max(_dist(shoulder_mid, hip_mid), 1e-3)

    rel_x = (palm[0] - shoulder_mid[0]) / shoulder_width
    rel_y = (palm[1] - shoulder_mid[1]) / torso_height

    # A hand hanging by the hip is a resting pose, not a signing contact.
    if rel_y > 0.60:
        return None

    # Shoulders / neck / face are point contacts.
    d_shoulder = min(_dist2(palm, ls), _dist2(palm, rs)) / shoulder_width
    if d_shoulder < 0.22:
        return "hamtouch"
    if d_shoulder < 0.42:
        return "hamclose"

    d_face = _dist2(palm, nose) / shoulder_width
    if d_face < 0.28:
        return "hamtouch"
    if d_face < 0.48:
        return "hamclose"

    # Hand inside the torso silhouette (in front of it for a frontal signer).
    if abs(rel_x) < 0.80 and -0.20 < rel_y < 0.60:
        return "hamclose"

    return None


def _best_body_contact(hands, pose_pts):
    """Strongest body contact across the visible hands (touch beats close)."""
    best = None
    for pts in hands:
        label = _body_contact(_palm_center(pts), pose_pts)
        if label is None:
            continue
        rank = 0 if label == "hamtouch" else 1
        if best is None or rank < best[0]:
            best = (rank, label)
    return best[1] if best else None



# ==========================================================
# FRAME PIPELINE
# ==========================================================

def _smooth(labels, window=5):
    smoothed = []
    n = len(labels)
    for i in range(n):
        s = max(0, i - window // 2)
        e = min(n, i + window // 2 + 1)
        valid = [v for v in labels[s:e] if v and v != NO_CONTACT]
        smoothed.append(Counter(valid).most_common(1)[0][0] if valid else NO_CONTACT)
    return smoothed


def run_contact_type_module(video_path, min_detection_confidence=0.4,
                            min_tracking_confidence=0.4,
                            use_body_contact=True):
    """
    Generalized contact-type module.
    Returns:
        dict: {"per_frame": [...], "final": <label>}
    """
    if not os.path.exists(video_path):
        print(f"[Contact] Video not found: {video_path}")
        return {"per_frame": [], "final": NO_CONTACT}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Contact] Failed to open video: {video_path}")
        return {"per_frame": [], "final": NO_CONTACT}

    hands_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    pose_detector = None
    if use_body_contact:
        pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    a_hist, b_hist = deque(maxlen=6), deque(maxlen=6)
    order_history = deque(maxlen=8)
    raw = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_res = hands_detector.process(rgb)
        pose_res = pose_detector.process(rgb) if pose_detector is not None else None

        label = None
        hands = []
        if hand_res.multi_hand_landmarks:
            hands = [_hand_points(hl, w, h) for hl in hand_res.multi_hand_landmarks]
        pose_pts = _pose_points(
            pose_res.pose_landmarks if pose_res is not None else None, w, h)

        # MediaPipe does not order hands, so pick the two largest palms and
        # only treat them as interacting when their fingertips are close.
        hands.sort(key=_palm_scale, reverse=True)

        if len(hands) >= 2:
            a, b = hands[0], hands[1]
            scale = 0.5 * (_palm_scale(a) + _palm_scale(b))
            min_tip = min(_dist(ta, tb) for ta in _tips(a) for tb in _tips(b))
            if min_tip < 1.8 * scale:
                pa, pb = _palm_center(a), _palm_center(b)
                order_history.append(1 if pa[0] < pb[0] else -1)
                a_hist.append(pa)
                b_hist.append(pb)
                label = _two_hand_label(a, b, a_hist, b_hist, order_history)
            else:
                order_history.append(0)
                a_hist.clear()
                b_hist.clear()
        else:
            order_history.append(0)
            a_hist.clear()
            b_hist.clear()

        if label is None:
            label = _best_body_contact(hands, pose_pts)

        raw.append(label if label else NO_CONTACT)

    cap.release()
    hands_detector.close()
    if pose_detector is not None:
        pose_detector.close()

    per_frame = _smooth(raw)
    valid = [v for v in per_frame if v != NO_CONTACT]
    final = Counter(valid).most_common(1)[0][0] if valid else NO_CONTACT
    return {"per_frame": per_frame, "final": final}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run generalized contact-type module.")
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()
    print(run_contact_type_module(args.video_path))
