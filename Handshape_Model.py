#!/usr/bin/env python
# coding: utf-8
"""
Generalized & Optimized Handshape Classification Module for HamNoSys.

Key Generalizations:
1. Rotation-Invariant Finger State Detection:
   - Replaced naive `lm[tip].y < lm[pip].y` (which failed whenever hands were tilted,
     sideways, pointing downwards, or moving) with coordinate-frame invariant joint
     bending angles and radial extension from the wrist/palm.
2. Scale-Invariant Metrics:
   - All spatial distances are normalized by hand palm scale (wrist to middle MCP distance),
     making classification invariant to camera distance, zoom, and video resolution.
3. Rotation-Invariant Thumb State:
   - Thumb abduction/extension measured relative to index MCP and palm plane rather than raw Y-axis.
4. Robust Detection & Tracking:
   - Uses adaptive confidence thresholds (defaulting to 0.4 instead of 0.6) to avoid dropping
     hands during fast signing motions or motion blur.
   - Supports dominant hand selection when multiple hands are in frame.
5. Temporal Filtering & Consistency:
   - Short rolling majority voting filters out single-frame jitter and detection flickers.
   - Corrected unreachable branches (e.g. `hamfingerstraightmod` vs `hamflathand`).
"""

import sys
import os
import cv2
import numpy as np
from collections import Counter, deque

# Import mediapipe or compatibility shim if available
try:
    import mediapipe as mp
except ImportError:
    mp = None

# Ensure solutions compat if running on modern mediapipe / Python 3.13
_this_dir = os.path.dirname(os.path.abspath(__file__))
_integration_dir = os.path.join(os.path.dirname(_this_dir), "Integration")
if os.path.exists(_integration_dir) and _integration_dir not in sys.path:
    sys.path.insert(0, _integration_dir)

try:
    import mediapipe_solutions_compat  # noqa: F401
except ImportError:
    pass


# =============================
# HAMNOSYS HANDSHAPE LABELS
# =============================

NODE_A = [
    "hamfist", "hamflathand", "hamfinger2", "hamfinger23", "hamfinger23spread",
    "hamfinger2345", "hampinch12", "hampinch12open", "hampinchall",
    "hamcee12", "hamceeall", "hamceeopen"
]

NODE_C = ["hamthumboutmod", "hamthumbacrossmod", "hamthumbopenmod"]

NODE_D = [
    "hamdoublebent", "hamdoublehooked",
    "hamfingerstraightmod", "hamfingerbendmod", "hamfingerhookmod"
]

NODE_E = [
    "hamthumb", "hamindexfinger", "hammiddlefinger", "hamringfinger", "hampinky",
    "hambetween", "hamfingernail", "hamfingerpad", "hamfingerside", "hamfingermidjoint"
]


# ==========================================================
# GEOMETRIC UTILITIES (ROTATION & SCALE INVARIANT)
# ==========================================================

def _lm_to_np(lm, idx):
    """Convert a landmark to 3D numpy array [x, y, z]."""
    return np.array([lm[idx].x, lm[idx].y, lm[idx].z], dtype=np.float32)


def _distance(p1, p2):
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(p1 - p2))


def _joint_angle(p_prev, p_curr, p_next):
    """
    Compute angle at p_curr in degrees (0 to 180).
    Straight extension = ~180 degrees. Full bend = < 90 degrees.
    """
    v1 = p_prev - p_curr
    v2 = p_next - p_curr
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cosine = np.dot(v1, v2) / (n1 * n2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def compute_palm_scale(lm):
    """
    Reference hand size: distance between wrist (0) and middle MCP (9).
    Falls back to index MCP (5) if needed.
    """
    wrist = _lm_to_np(lm, 0)
    middle_mcp = _lm_to_np(lm, 9)
    scale = _distance(wrist, middle_mcp)
    return max(scale, 1e-4)


# =============================
# ROTATION-INVARIANT FINGER STATE
# =============================

def is_finger_extended(mcp_idx, pip_idx, dip_idx, tip_idx, lm, palm_scale):
    """
    Determine whether a finger (index, middle, ring, pinky) is extended.
    Invariant to 3D hand rotation, tilt, and distance from camera.
    Uses joint angles and relative distance from wrist and MCP.
    """
    wrist = _lm_to_np(lm, 0)
    mcp   = _lm_to_np(lm, mcp_idx)
    pip   = _lm_to_np(lm, pip_idx)
    dip   = _lm_to_np(lm, dip_idx)
    tip   = _lm_to_np(lm, tip_idx)

    # 1. Joint angles: PIP angle and DIP angle
    pip_angle = _joint_angle(mcp, pip, dip)
    dip_angle = _joint_angle(pip, dip, tip)

    # 2. Distance from wrist: extended finger has tip significantly farther from wrist than PIP
    dist_wrist_tip = _distance(wrist, tip)
    dist_wrist_pip = _distance(wrist, pip)

    # 3. Distance from MCP: curled finger folds back towards MCP
    dist_mcp_tip = _distance(mcp, tip)
    dist_mcp_pip = _distance(mcp, pip)

    # Composite decision:
    # If the finger is curled (fist/claw), pip_angle < 120 or tip is closer to MCP than PIP
    is_straight = (pip_angle > 125.0 and dip_angle > 120.0)
    dist_ratio = dist_wrist_tip / max(dist_wrist_pip, 1e-4)

    if dist_ratio > 1.25 and is_straight:
        return True
    if dist_ratio < 1.05 or dist_mcp_tip < dist_mcp_pip * 1.05:
        return False

    # Intermediate / borderline cases:
    return pip_angle > 135.0


def is_thumb_extended(lm, palm_scale):
    """
    Determine whether thumb is extended away from palm.
    Thumb articulates radially/laterally. We measure:
    1. Distance from thumb tip (4) to index MCP (5) normalized by palm scale.
    2. Distance from thumb tip (4) to wrist (0) compared to IP (3).
    3. Angle at thumb MCP (2) and IP (3).
    """
    wrist = _lm_to_np(lm, 0)
    cmc   = _lm_to_np(lm, 1)
    mcp   = _lm_to_np(lm, 2)
    ip    = _lm_to_np(lm, 3)
    tip   = _lm_to_np(lm, 4)
    idx_mcp = _lm_to_np(lm, 5)

    # Distance to index MCP normalized by palm scale
    dist_idx_mcp = _distance(tip, idx_mcp) / palm_scale

    # Thumb extension relative to wrist vs IP
    dist_wrist_tip = _distance(wrist, tip)
    dist_wrist_ip  = _distance(wrist, ip)

    # Angle at IP
    ip_angle = _joint_angle(mcp, ip, tip)

    # Thumb is extended if tip is far from index MCP and not folded against palm
    if dist_idx_mcp > 0.55 and ip_angle > 125.0 and dist_wrist_tip >= dist_wrist_ip:
        return True
    elif dist_idx_mcp < 0.38:
        return False

    return dist_idx_mcp > 0.45


def finger_open(tip, pip, lm):
    """
    Backward-compatible wrapper for finger_open.
    Dynamically routes to rotation-invariant check.
    """
    palm_scale = compute_palm_scale(lm)
    mapping = {
        4: (1, 2, 3, 4),
        8: (5, 6, 7, 8),
        12: (9, 10, 11, 12),
        16: (13, 14, 15, 16),
        20: (17, 18, 19, 20)
    }
    if tip == 4:
        return is_thumb_extended(lm, palm_scale)
    if tip in mapping:
        m, p, d, t = mapping[tip]
        return is_finger_extended(m, p, d, t, lm, palm_scale)
    # Fallback
    return lm[tip].y < lm[pip].y


def get_finger_states(lm):
    """
    Returns boolean state (extended vs folded) for all 5 fingers:
    (thumb, index, middle, ring, pinky)
    Rotation-invariant and scale-invariant.
    """
    palm_scale = compute_palm_scale(lm)
    thumb  = is_thumb_extended(lm, palm_scale)
    index  = is_finger_extended(5, 6, 7, 8, lm, palm_scale)
    middle = is_finger_extended(9, 10, 11, 12, lm, palm_scale)
    ring   = is_finger_extended(13, 14, 15, 16, lm, palm_scale)
    pinky  = is_finger_extended(17, 18, 19, 20, lm, palm_scale)

    return thumb, index, middle, ring, pinky


# =============================
# HANDSHAPE CLASSIFICATION
# =============================

def classify_handshape(lm):
    """
    Classify Handshape into HamNoSys label.
    Supports comprehensive rules, handles edge cases, and removes unreachable branches.
    """
    palm_scale = compute_palm_scale(lm)
    thumb, index, middle, ring, pinky = get_finger_states(lm)

    # Fingertip coordinates for pinch & spread detection
    t_tip = _lm_to_np(lm, 4)
    i_tip = _lm_to_np(lm, 8)
    m_tip = _lm_to_np(lm, 12)

    thumb_index_dist = _distance(t_tip, i_tip) / palm_scale
    index_middle_dist = _distance(i_tip, m_tip) / palm_scale

    # 1. FIST: All fingers folded
    if not thumb and not index and not middle and not ring and not pinky:
        return "hamfist"

    # 2. PINCH 12: Thumb and Index tips touching / very close
    if thumb_index_dist < 0.30 and not middle and not ring and not pinky:
        return "hampinch12"

    # 3. C-SHAPE 12: Thumb and Index curved towards each other (open pinch or c-shape)
    if 0.30 <= thumb_index_dist < 0.70 and not middle and not ring and not pinky and (index or thumb):
        return "hamcee12"

    # 4. FLAT HAND (all 4 fingers extended + thumb open)
    if index and middle and ring and pinky and thumb:
        return "hamflathand"

    # 5. STRAIGHT FINGERS (all 4 fingers extended, thumb folded/tucked)
    if index and middle and ring and pinky and not thumb:
        return "hamfingerstraightmod"

    # 6. TWO FINGERS (Index + Middle)
    if index and middle and not ring and not pinky:
        if index_middle_dist > 0.40:
            return "hamfinger23spread"
        return "hamfinger23"

    # 7. INDEX FINGER ONLY
    if index and not middle and not ring and not pinky:
        return "hamindexfinger"

    # 8. THUMB OUT MODIFIER (Thumbs up / thumb extended while fingers folded)
    if thumb and not index and not middle and not ring and not pinky:
        return "hamthumboutmod"

    # 9. FOUR FINGERS SPREAD / OPEN (without pinky folded, etc.)
    if index and middle and ring and not pinky:
        return "hamfinger2345"

    # 10. Fallbacks for individual prominent fingers
    if index:
        return "hamindexfinger"
    if middle:
        return "hammiddlefinger"
    if pinky and not index and not middle and not ring:
        return "hampinky"
    if thumb:
        return "hamthumb"
    if ring:
        return "hamringfinger"

    return "hamfist"


# =============================
# TEMPORAL SMOOTHING HELPER
# =============================

def smooth_predictions(raw_labels, window_size=5):
    """
    Applies sliding window majority voting to remove transient flickers and 'none' dropouts.
    """
    smoothed = []
    n = len(raw_labels)
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        sub = [l for l in raw_labels[start:end] if l not in (None, "none", "")]
        if sub:
            smoothed.append(Counter(sub).most_common(1)[0][0])
        else:
            smoothed.append(raw_labels[i] if raw_labels[i] is not None else "none")
    return smoothed


# =============================
# HANDSHAPE MODULE ENTRYPOINT
# =============================

def run_handshape_module(video_path, min_detection_confidence=0.4, min_tracking_confidence=0.4):
    """
    Generalized Handshape Module.
    Works across varying video qualities, framerates, resolutions, and hand angles.
    Returns:
        dict: {"per_frame": [...], "final": <dominant_label>}
    """
    if not os.path.exists(video_path):
        print(f"[Handshape] Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Handshape] Failed to open video: {video_path}")
        return None

    # Instantiate local MediaPipe Hands instance to avoid global state / thread leaks
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # detect up to 2 hands to handle two-handed gestures or pick dominant hand
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

    raw_labels = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(img)

        label = "none"

        if results.multi_hand_landmarks:
            # Pick the primary/dominant hand (the one with largest bounding box / palm scale)
            best_lm = None
            max_scale = -1.0
            for hand_landmarks in results.multi_hand_landmarks:
                scale = compute_palm_scale(hand_landmarks.landmark)
                if scale > max_scale:
                    max_scale = scale
                    best_lm = hand_landmarks.landmark

            if best_lm is not None:
                label = classify_handshape(best_lm)

        raw_labels.append(label)

    cap.release()
    try:
        detector.close()
    except Exception:
        pass

    # Temporal smoothing
    smoothed_labels = smooth_predictions(raw_labels, window_size=5)

    # Determine final dominant label (excluding "none" unless all frames are "none")
    valid = [l for l in smoothed_labels if l not in (None, "none", "")]
    final_label = Counter(valid).most_common(1)[0][0] if valid else "none"

    return {
        "per_frame": smoothed_labels,
        "final": final_label
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run generalized Handshape classification.")
    parser.add_argument("video_path", nargs="?", default="Prompt_1.mp4", help="Path to video file")
    args = parser.parse_args()

    res = run_handshape_module(args.video_path)
    if res:
        print(f"Final Handshape Label: {res.get('final')}")
        print(f"Total Frames: {len(res.get('per_frame', []))}")
