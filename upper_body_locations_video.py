#!/usr/bin/env python
# coding: utf-8
"""
Generalized Upper-Body Location Module for HamNoSys.

Fixes for cross-video consistency:
1. Actually uses hand landmarks (palm centre) instead of estimating the hand
   from the pose wrist with integer division.
2. Every threshold is normalized by the signer's shoulder width, torso height
   or limb length, so framing, zoom and body size no longer change labels.
3. Arms are detected with point-to-segment distance, which works when arms are
   horizontal or diagonal (the old y-range rule only worked for hanging arms).
4. Torso labels require the hand to be inside the torso silhouette, so hands
   resting beside the body are no longer reported as hamchest/hamstomach.
5. Both hands are evaluated and the closest/most specific hand-body contact
   wins, instead of always preferring the left hand.
6. Pose visibility is checked and output is one label per video frame.

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


def _xyz(lm, w, h):
    return np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float32)


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _dist2(a, b):
    """Body-plane distance. Pose and hand landmarks estimate z in different
    reference frames, so cross-model distances must stay in x/y."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _point_segment_distance(p, a, b):
    ab = b - a
    denom = float(np.dot(ab[:2], ab[:2]))
    if denom < 1e-6:
        return _dist2(p, a)
    t = float(np.dot(p[:2] - a[:2], ab[:2]) / denom)
    t = max(0.0, min(1.0, t))
    return float(np.hypot(p[0] - (a[0] + t * ab[0]), p[1] - (a[1] + t * ab[1])))


def _pose_xyz(pose_landmarks, w, h):
    if pose_landmarks is None:
        return None
    return np.array([_xyz(lm, w, h) for lm in pose_landmarks.landmark], dtype=np.float32)


def _pose_visible(pose_landmarks, indices, threshold=0.25):
    if pose_landmarks is None:
        return False
    lms = pose_landmarks.landmark
    return all(getattr(lms[i], "visibility", 1.0) >= threshold for i in indices)


def _hand_center(hand_landmarks, w, h):
    pts = np.array([_xyz(lm, w, h) for lm in hand_landmarks.landmark], dtype=np.float32)
    return pts.mean(axis=0)


def _classify_hand(contact, pts, side, from_pose=False, arm_ok=True):
    """
    Classify one hand position against a normalized body frame.
    Returns (label, score) where a smaller score is a tighter match.
    A pose wrist lies on the arm itself, so the arm-segment rules are skipped
    for it; only shoulder/torso locations are meaningful. The arm rules also
    need both that arm's elbow and wrist to be tracked.
    """
    ls, rs = pts[11], pts[12]
    le, re = pts[13], pts[14]
    lw, rw = pts[15], pts[16]
    lh, rh = pts[23], pts[24]

    shoulder_mid = (ls + rs) * 0.5
    hip_mid = (lh + rh) * 0.5
    shoulder_width = max(_dist(ls, rs), 1e-3)
    torso_height = max(_dist(shoulder_mid, hip_mid), 1e-3)
    torso_center_x = shoulder_mid[0]
    shoulder_y = shoulder_mid[1]
    hip_y = hip_mid[1]

    cx, cy = contact[0], contact[1]
    lateral = abs(cx - torso_center_x) / shoulder_width

    # 1. Shoulder region
    d_shoulder = min(_dist2(contact, ls), _dist2(contact, rs)) / shoulder_width
    if d_shoulder < 0.42:
        label = "hamshouldertop" if cy < shoulder_y - 0.05 * torso_height else "hamshoulders"
        return label, d_shoulder

    # 2. Elbow region
    elbow = le if side == "L" else re
    upper_arm_len = max(_dist2(ls if side == "L" else rs, elbow), 1e-3)
    if not from_pose and arm_ok:
        d_elbow = _dist2(contact, elbow) / upper_arm_len
        if d_elbow < 0.25:
            return "hamelbowinside", d_elbow

        # 3. Along the arm (works for any arm orientation)
        shoulder = ls if side == "L" else rs
        wrist = lw if side == "L" else rw
        forearm_len = max(_dist2(elbow, wrist), 1e-3)

        d_upper = _point_segment_distance(contact, shoulder, elbow) / upper_arm_len
        d_fore = _point_segment_distance(contact, elbow, wrist) / forearm_len

        if d_upper < 0.28 and lateral > 0.18:
            return "hamupperarm", d_upper
        if d_fore < 0.28 and lateral > 0.18:
            return "hamlowerarm", d_fore

        # 4. Back of the wrist
        d_wrist = _dist2(contact, wrist) / forearm_len
        if d_wrist < 0.30 and d_elbow > 0.55:
            return "hamwristback", d_wrist

    # 5. Torso: require the hand to be inside the body silhouette
    inside = (shoulder_y - 0.10 * torso_height < cy < hip_y + 0.10 * torso_height
              and lateral < 0.55)
    if inside:
        rel = (cy - shoulder_y) / torso_height
        if rel < 0.33:
            return "hamchest", 0.30 + lateral
        if rel < 0.70:
            return "hamstomach", 0.35 + lateral
        return "hambelowstomach", 0.40 + lateral

    return None, 1e9


_PRIORITY = {
    "hamshouldertop": 0, "hamshoulders": 0, "hamelbowinside": 1,
    "hamwristback": 2, "hamupperarm": 3, "hamlowerarm": 3,
    "hamchest": 4, "hamstomach": 4, "hambelowstomach": 4,
}


def _classify_frame(pose_landmarks, hand_res, w, h):
    pts = _pose_xyz(pose_landmarks, w, h)
    if pts is None or not _pose_visible(pose_landmarks, [11, 12, 23, 24], 0.20):
        return None

    lms = pose_landmarks.landmark
    vis = [getattr(lm, "visibility", 1.0) for lm in lms]
    arm_ok = {"L": vis[13] >= 0.20 and vis[15] >= 0.20,
              "R": vis[14] >= 0.20 and vis[16] >= 0.20}

    # Candidate contact points: real hand landmarks plus pose wrists. The hand
    # detector often drops a shaped hand (fist, "O"), while the pose wrist is
    # still tracked, so both are evaluated.
    candidates = []
    shoulder_mid_x = (pts[11][0] + pts[12][0]) * 0.5
    shoulder_width = max(_dist(pts[11], pts[12]), 1e-3)
    if hand_res is not None and hand_res.multi_hand_landmarks:
        for hand_lm in hand_res.multi_hand_landmarks:
            center = _hand_center(hand_lm, w, h)
            # MediaPipe handedness assumes a mirrored selfie frame and flips on
            # third-person recordings, so derive the side from the body instead.
            side = "L" if center[0] > shoulder_mid_x else "R"
            candidates.append((center, side, False))
    for wrist, side in ((pts[15], "L"), (pts[16], "R")):
        if all(_dist2(wrist, c) > 0.45 * shoulder_width for c, _, _ in candidates):
            candidates.append((wrist, side, True))

    best_label = None
    best_key = None
    for contact, side, from_pose in candidates:
        label, score = _classify_hand(contact, pts, side,
                                      from_pose=from_pose, arm_ok=arm_ok[side])
        if label is None:
            continue
        key = (_PRIORITY.get(label, 9), score)
        if best_key is None or key < best_key:
            best_key = key
            best_label = label
    return best_label


def _smooth(values, window=5):
    smoothed = []
    n = len(values)
    for i in range(n):
        s = max(0, i - window // 2)
        e = min(n, i + window // 2 + 1)
        valid = [v for v in values[s:e] if v]
        smoothed.append(Counter(valid).most_common(1)[0][0] if valid else values[i])
    return smoothed


def run_upper_body_location_module(video_path, min_detection_confidence=0.4,
                                   min_tracking_confidence=0.4):
    """
    Generalized upper-body location module.
    Returns:
        dict: {"per_frame": [...], "final": <label> | None}
    """
    if not os.path.exists(video_path):
        print(f"[Upper Body] Video not found: {video_path}")
        return {"per_frame": [], "final": None}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Upper Body] Failed to open video: {video_path}")
        return {"per_frame": [], "final": None}

    pose_detector = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
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
        pose_res = pose_detector.process(rgb)
        hand_res = hands_detector.process(rgb)
        try:
            label = _classify_frame(pose_res.pose_landmarks, hand_res, w, h)
        except Exception:
            label = None
        raw.append(label)

    cap.release()
    pose_detector.close()
    hands_detector.close()

    per_frame = _smooth(raw)
    valid = [v for v in per_frame if v]
    final = Counter(valid).most_common(1)[0][0] if valid else None
    return {"per_frame": per_frame, "final": final}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run generalized upper-body location module.")
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()
    print(run_upper_body_location_module(args.video_path))
