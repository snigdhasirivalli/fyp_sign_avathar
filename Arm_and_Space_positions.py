#!/usr/bin/env python
# coding: utf-8
"""
Generalized Arm & Space Location Module for HamNoSys.

Follows the documented HamNoSys definitions but removes video-specific units:
- Arm straightness R = |S-W| / (|S-E| + |E-W|), independent of resolution.
- Elbow angle from the 3D cosine rule.
- Lateral and depth offsets divided by the signer's shoulder width instead of
  raw normalized-frame constants (which only worked for one camera framing).
- Torso height normalizes the "beside the body" vertical band.
- The more active arm is selected instead of always trusting the farther hand.
- Pose visibility is checked and every frame gets an aligned label.

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


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return 180.0
    cosang = float(np.dot(ba, bc) / denom)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def _pose_points(pose_landmarks, w, h):
    if pose_landmarks is None:
        return None
    return np.array([_xyz(lm, w, h) for lm in pose_landmarks.landmark], dtype=np.float32)


def _pose_visible(pose_landmarks, indices, threshold=0.25):
    if pose_landmarks is None:
        return False
    lms = pose_landmarks.landmark
    return all(getattr(lms[i], "visibility", 1.0) >= threshold for i in indices)


def _arm_features(pts, side):
    if side == "L":
        s, e, w = pts[11], pts[13], pts[15]
    else:
        s, e, w = pts[12], pts[14], pts[16]

    upper = max(_dist(s, e), 1e-4)
    fore = max(_dist(e, w), 1e-4)
    straight = _dist(s, w) / (upper + fore)          # 0..1, 1 = fully straight
    elbow = _angle(s, e, w)

    shoulder_mid = (pts[11] + pts[12]) * 0.5
    hip_mid = (pts[23] + pts[24]) * 0.5
    shoulder_width = max(_dist(pts[11], pts[12]), 1e-3)
    torso_height = max(_dist(shoulder_mid, hip_mid), 1e-3)

    return {
        "straight": straight,
        "elbow": elbow,
        "lateral": abs(w[0] - shoulder_mid[0]) / shoulder_width,
        "wrist_height": (w[1] - shoulder_mid[1]) / torso_height,
    }


def classify_arm_space(pts):
    """Classify one frame from pose landmarks. Returns (label, score)."""
    fL = _arm_features(pts, "L")
    fR = _arm_features(pts, "R")

    # 1. Both elbows clearly bent in the signing space.
    if (fL["elbow"] < 120.0 and fR["elbow"] < 120.0
            and max(fL["wrist_height"], fR["wrist_height"]) < 0.55):
        return "hamdoublebent", 0.0

    # 2. The signing arm is the raised one; the other usually hangs by the hip.
    side = "L" if fL["wrist_height"] < fR["wrist_height"] else "R"
    f = fL if side == "L" else fR

    # 3. Resting pose: the lower/straighter arm still hangs by the body.
    if f["wrist_height"] > 0.60 and f["elbow"] > 150.0:
        return None, 1e9

    # 4. Fully extended arm.
    if f["elbow"] > 155.0 and f["straight"] > 0.92 and f["wrist_height"] < 0.55:
        return "hamarmextended", f["straight"]

    # 5. Hand raised to head / shoulder height -> beside the head.
    if f["wrist_height"] < -0.10:
        return "hamlrat", abs(f["wrist_height"])

    # 6. Hand held near the body midline at chest/stomach height.
    if f["lateral"] < 0.18:
        return "hamlrat", f["lateral"]

    # 7. Hand held out at chest height on the signing side.
    if f["lateral"] < 0.80:
        return "hamlrbeside", f["lateral"]

    return "hamneutralspace", f["lateral"]



def _smooth(values, window=5):
    smoothed = []
    n = len(values)
    for i in range(n):
        s = max(0, i - window // 2)
        e = min(n, i + window // 2 + 1)
        valid = [v for v in values[s:e] if v]
        smoothed.append(Counter(valid).most_common(1)[0][0] if valid else values[i])
    return smoothed


def run_arm_space_module(video_path, min_detection_confidence=0.4,
                         min_tracking_confidence=0.4):
    """
    Generalized arm & space module.
    Returns:
        dict: {"per_frame": [...], "final": <label> | None}
    """
    if not os.path.exists(video_path):
        print(f"[Arm/Space] Video not found: {video_path}")
        return {"per_frame": [], "final": None}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Arm/Space] Failed to open video: {video_path}")
        return {"per_frame": [], "final": None}

    pose_detector = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
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

        label = None
        if _pose_visible(pose_res.pose_landmarks, [11, 12, 13, 14, 15, 16, 23, 24], 0.20):
            pts = _pose_points(pose_res.pose_landmarks, w, h)
            try:
                label, _ = classify_arm_space(pts)
            except Exception:
                label = None
        raw.append(label)

    cap.release()
    pose_detector.close()

    per_frame = _smooth(raw)
    valid = [v for v in per_frame if v]
    final = Counter(valid).most_common(1)[0][0] if valid else None
    return {"per_frame": per_frame, "final": final}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run generalized arm & space module.")
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()
    print(run_arm_space_module(args.video_path))
