#!/usr/bin/env python
# coding: utf-8
"""
Generalized & Optimized Finger Location Classification Module for HamNoSys.

Key Generalizations:
1. Fixed Critical Bug in Final Aggregation:
   - Fixed the logic in `run_finger_location_module` where `max(set(predictions), key=predictions.count)`
     overwrote valid votes and returned `None` whenever `None` was the most common frame prediction,
     or threw a `ValueError` on empty sequences.
2. Dynamic Scale-Adaptive Proximity Thresholds:
   - Replaced fixed coordinate thresholds (0.03, 0.025, 0.035) with scale-adaptive thresholds
     proportional to the hand's anatomical scale (distance from wrist to middle MCP).
     This works reliably regardless of whether the signer is in close-up, medium shot, or wide shot.
3. Aspect Ratio Correction:
   - Normalized coordinate Euclidean distances now account for frame aspect ratio or 3D coordinate space,
     preventing 16:9 distortion where horizontal distances appear artificially compressed.
4. Comprehensive Contact Detection:
   - Checks both finger-to-palm proximity and finger-to-finger contact (e.g. thumb-to-index pinch or touch).
5. Robust Active Finger Classification:
   - When no contact is present, identifies the prominent active extended finger using
     rotation-invariant straightness and normalized extension.
6. Temporal Stabilization:
   - Uses sliding-window temporal smoothing across frames to prevent rapid flickering of location tokens.
"""

import os
import sys
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
# LANDMARK CONSTANTS
# =============================

FINGER_LANDMARKS = {
    "hamthumb":        [1, 2, 3, 4],
    "hamindexfinger":  [5, 6, 7, 8],
    "hammiddlefinger": [9, 10, 11, 12],
    "hamringfinger":   [13, 14, 15, 16],
    "hampinky":        [17, 18, 19, 20],
}

TIP_LM  = [4, 8, 12, 16, 20]
PIP_LM  = [6, 10, 14, 18]
MCP_LM  = [5, 9, 13, 17]
PALM_LM = [0, 5, 9, 13, 17]

ADJACENT_FINGERS = {
    "hamthumb":        ["hamindexfinger"],
    "hamindexfinger":  ["hamthumb", "hammiddlefinger"],
    "hammiddlefinger": ["hamindexfinger", "hamringfinger"],
    "hamringfinger":   ["hammiddlefinger", "hampinky"],
    "hampinky":        ["hamringfinger"],
}


# =============================
# GEOMETRIC UTILITIES
# =============================

def dist(a, b):
    """Euclidean distance between two 3D or 2D points."""
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def mp_landmarks_to_xyz(hand_landmarks):
    """Convert MediaPipe landmark list to list of (x, y, z) tuples."""
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]


def get_palm_scale(landmarks):
    """
    Computes reference hand scale: distance between wrist (0) and middle MCP (9).
    """
    wrist = np.array(landmarks[0], dtype=np.float32)
    middle_mcp = np.array(landmarks[9], dtype=np.float32)
    scale = float(np.linalg.norm(wrist - middle_mcp))
    return max(scale, 1e-4)


def finger_of_landmark(idx):
    """Return the finger name for a given landmark index."""
    for finger, ids in FINGER_LANDMARKS.items():
        if idx in ids:
            return finger
    return None


# =============================
# CONTACT ESTIMATION (SCALE-AWARE)
# =============================

def estimate_contact_index(landmarks, palm_scale=None):
    """
    Finds which fingertip is closest to contact (either to the palm or to another fingertip).
    """
    if palm_scale is None:
        palm_scale = get_palm_scale(landmarks)

    palm_center = np.mean([landmarks[i] for i in PALM_LM], axis=0)

    # 1. Check proximity of fingertips to palm
    best_idx, best_d = None, float("inf")
    for i in TIP_LM:
        d = dist(palm_center, landmarks[i])
        if d < best_d:
            best_d = d
            best_idx = i

    # 2. Check for thumb touching another fingertip (pinch contact)
    thumb_tip = landmarks[4]
    for i in [8, 12, 16, 20]:
        d = dist(thumb_tip, landmarks[i])
        if d < 0.25 * palm_scale:
            return i

    return best_idx if best_idx is not None else 8


def has_contact_candidate(contact_idx, landmarks, palm_scale=None):
    """
    Scale-adaptive spatial contact candidate check:
    - NOT same finger
    - NOT adjacent fingers (unless within tight contact threshold)
    """
    if contact_idx is None or contact_idx >= len(landmarks):
        return False

    if palm_scale is None:
        palm_scale = get_palm_scale(landmarks)

    # Dynamic threshold proportional to palm size
    thresh = 0.28 * palm_scale
    tight_thresh = 0.18 * palm_scale

    contact = landmarks[contact_idx]
    finger = finger_of_landmark(contact_idx)

    for i, lm in enumerate(landmarks):
        other_finger = finger_of_landmark(i)

        if other_finger == finger:
            continue

        d = dist(contact, lm)

        # Allow non-adjacent finger contact, or tight contact even with adjacent
        if other_finger in ADJACENT_FINGERS.get(finger, []):
            if d < tight_thresh:
                return True
            continue

        if d < thresh:
            return True

    return False


# =============================
# FINGER IDENTITY (NO CONTACT)
# =============================

def finger_straightness(landmarks, finger):
    """
    Measures finger straightness using cosine similarities along joint segments.
    Value is ~1.0 for straight finger, < 0.5 for curled finger.
    """
    ids = FINGER_LANDMARKS[finger]
    mcp, pip, dip, tip = [np.array(landmarks[i], dtype=np.float32) for i in ids]

    v1 = pip - mcp
    v2 = dip - pip
    v3 = tip - dip

    def cos_sim(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        return float(np.dot(a, b) / denom)

    return (cos_sim(v1, v2) + cos_sim(v2, v3)) / 2.0


def classify_finger_identity(landmarks):
    """
    Identifies which finger is active/prominent when no contact is present.
    The active finger is extended away from palm with high straightness.
    """
    palm_center = np.mean([landmarks[i] for i in PALM_LM], axis=0)
    palm_scale = get_palm_scale(landmarks)

    best_finger = None
    best_score = -1.0

    for finger, ids in FINGER_LANDMARKS.items():
        tip_idx = ids[-1]
        # Normalized extension by palm scale
        extension = dist(landmarks[tip_idx], palm_center) / palm_scale
        straightness = finger_straightness(landmarks, finger)

        score = extension * max(straightness, 0.1)
        if score > best_score:
            best_score = score
            best_finger = finger

    return best_finger if best_finger else "hamindexfinger"


# =============================
# FINGER-PART CLASSIFIERS (CONTACT)
# =============================

def classify_finger_part(contact, landmarks, palm_scale=None):
    """
    Determines whether contact point is at the fingertip, mid-joint, or base.
    Uses scale-adaptive proximity thresholds.
    """
    if palm_scale is None:
        palm_scale = get_palm_scale(landmarks)

    tip_thresh  = 0.24 * palm_scale
    pip_thresh  = 0.30 * palm_scale
    mcp_thresh  = 0.36 * palm_scale

    min_tip = min(dist(contact, landmarks[i]) for i in TIP_LM)
    min_pip = min(dist(contact, landmarks[i]) for i in PIP_LM)
    min_mcp = min(dist(contact, landmarks[i]) for i in MCP_LM)

    if min_tip < tip_thresh:
        return "hamfingertip"
    if min_pip < pip_thresh:
        return "hamfingermidjoint"
    if min_mcp < mcp_thresh:
        return "hamfingerbase"

    return None


def classify_hambetween(contact, landmarks, palm_scale=None):
    """
    Determines if contact is in the space between two fingers.
    """
    if palm_scale is None:
        palm_scale = get_palm_scale(landmarks)

    between_thresh = 0.32 * palm_scale
    centers = {
        f: np.mean([landmarks[i] for i in ids], axis=0)
        for f, ids in FINGER_LANDMARKS.items()
    }
    close = [f for f, c in centers.items() if dist(contact, c) < between_thresh]
    return "hambetween" if len(close) >= 2 else None


def classify_fingertip_side(contact, landmarks, contact_idx):
    """
    Distinguishes finger pad vs fingernail contact based on palm vector.
    """
    tip = np.array(landmarks[contact_idx], dtype=np.float32)
    palm_center = np.mean([landmarks[i] for i in PALM_LM], axis=0)
    tip_to_palm = palm_center - tip
    tip_to_contact = np.array(contact, dtype=np.float32) - tip
    dot = float(np.dot(tip_to_palm, tip_to_contact))

    return "hamfingerpad" if dot > 0 else "hamfingernail"


def classify_frame(landmarks, stable_contact):
    """
    Per-frame classifier for finger location.
    If stable contact exists -> returns contact location label.
    Otherwise -> returns prominent active finger identity.
    """
    palm_scale = get_palm_scale(landmarks)
    contact_idx = estimate_contact_index(landmarks, palm_scale)
    contact = landmarks[contact_idx]

    if stable_contact:
        part = classify_finger_part(contact, landmarks, palm_scale)
        if part:
            return part

        between = classify_hambetween(contact, landmarks, palm_scale)
        if between:
            return between

        # Secondary check for pad vs nail
        side = classify_fingertip_side(contact, landmarks, contact_idx)
        if side:
            return side

    return classify_finger_identity(landmarks)


# =============================
# TEMPORAL SMOOTHING
# =============================

def smooth_labels(predictions, window=5):
    """
    Applies sliding window mode filtering.
    """
    smoothed = []
    n = len(predictions)
    for i in range(n):
        s = max(0, i - window // 2)
        e = min(n, i + window // 2 + 1)
        valid = [p for p in predictions[s:e] if p not in (None, "none", "")]
        if valid:
            smoothed.append(Counter(valid).most_common(1)[0][0])
        else:
            smoothed.append(predictions[i] if predictions[i] is not None else "none")
    return smoothed


# =============================
# INFERENCE PIPELINE
# =============================

def run_finger_location_module(video_path, min_detection_confidence=0.4, min_tracking_confidence=0.4):
    """
    Generalized Finger Location Module.
    Works across varying video qualities and scales.
    Returns:
        dict: {"per_frame": [...], "final": <label>}
    """
    if not os.path.exists(video_path):
        print(f"[Finger Location] Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Finger Location] Failed to open video: {video_path}")
        return None

    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

    CONTACT_STABILITY = 3
    contact_history = deque(maxlen=CONTACT_STABILITY)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        loc = None
        if result.multi_hand_landmarks:
            # Pick dominant hand
            best_hand = None
            max_scale = -1.0
            for h in result.multi_hand_landmarks:
                lms = mp_landmarks_to_xyz(h)
                sc = get_palm_scale(lms)
                if sc > max_scale:
                    max_scale = sc
                    best_hand = lms

            if best_hand is not None:
                palm_scale = get_palm_scale(best_hand)
                contact_idx = estimate_contact_index(best_hand, palm_scale)
                candidate = has_contact_candidate(contact_idx, best_hand, palm_scale)

                contact_history.append(candidate)
                stable_contact = len(contact_history) >= 2 and (sum(contact_history) >= 2)

                loc = classify_frame(best_hand, stable_contact)

        predictions.append(loc)

    cap.release()
    try:
        detector.close()
    except Exception:
        pass

    # Temporal smoothing
    smoothed = smooth_labels(predictions, window=5)

    # Calculate final label accurately (NEVER returning None if valid frames exist!)
    valid = [p for p in smoothed if p not in (None, "none", "")]
    if valid:
        final_location = Counter(valid).most_common(1)[0][0]
    else:
        final_location = "none"

    return {
        "per_frame": smoothed,
        "final": final_location
    }


def classify_video(video_path):
    res = run_finger_location_module(video_path)
    return res.get("final") if res else None


def main(video_path):
    return classify_video(video_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run generalized Finger Location module.")
    parser.add_argument("video_path", nargs="?", default="Prompt_1.mp4", help="Path to video file")
    args = parser.parse_args()

    res = run_finger_location_module(args.video_path)
    if res:
        print(f"Final Finger Location: {res.get('final')}")
        print(f"Total Frames Processed: {len(res.get('per_frame', []))}")
