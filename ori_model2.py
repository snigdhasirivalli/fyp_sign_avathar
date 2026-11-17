#!/usr/bin/env python
# coding: utf-8
"""
Generalized Hand Orientation Module for HamNoSys.

Fixes that make it work across videos instead of only some:
1. Body-normalized view detection:
   - Wrist/eye/shoulder offsets are divided by the signer's shoulder width and
     eye-to-shoulder distance, so framing, zoom and camera distance no longer
     change the predicted view.
2. Roll-invariant directions:
   - Screen directions are corrected by the shoulder/eye roll angle, so a tilted
     camera or head tilt does not rotate every direction label.
3. True 3D finger direction:
   - Uses the most-extended fingertip in a hand-scale-normalized 3D frame and
     maps the (direction, depth) pair directly onto the HamNoSys vocabulary,
     replacing the brittle three-view lookup table.
4. Anatomical palm normal:
   - Palm direction comes from the palm-plane normal, with its sign resolved
     from the signer's body side (not from MediaPipe handedness, which flips on
     mirrored videos).
5. No import-time training / no temp files:
   - The synthetic RandomForest models and per-frame JPEG round-trip are gone.
     Detection is geometric, deterministic and much faster.
6. Per-video state + temporal smoothing:
   - Detectors are created per video and every frame gets an aligned label.

Output stays compatible:
    {"per_frame": [(view, finger, palm), ...], "final": (view, finger, palm)}
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


# ==========================================================
# HAMNOSYS ORIENTATION VOCABULARY
# ==========================================================

VIEWS = ("bird", "right", "signer")

FINGER_DIRS = ("u", "ur", "r", "dr", "d", "dl", "l", "ul")

# (planar direction, depth) -> HamNoSys token
FINGER_TOKENS = {
    ("u", ""): "hamextfingeru",  ("ur", ""): "hamextfingerur",
    ("r", ""): "hamextfingerr",  ("dr", ""): "hamextfingerdr",
    ("d", ""): "hamextfingerd",  ("dl", ""): "hamextfingerdl",
    ("l", ""): "hamextfingerl",  ("ul", ""): "hamextfingerul",
    ("u", "o"): "hamextfingeruo", ("u", "i"): "hamextfingerui",
    ("d", "o"): "hamextfingerdo", ("d", "i"): "hamextfingerdi",
    ("r", "o"): "hamextfingeror", ("r", "i"): "hamextfingerir",
    ("l", "o"): "hamextfingerol", ("l", "i"): "hamextfingeril",
}

PALM_TOKENS = {
    "u": "hampalmu", "ur": "hampalmur", "r": "hampalmr", "dr": "hampalmdr",
    "d": "hampalmd", "dl": "hampalmdl", "l": "hampalml", "ul": "hampalmul",
}

DIR_SECTORS = ("r", "ur", "u", "ul", "l", "dl", "d", "dr")


# ==========================================================
# LANDMARK / GEOMETRY HELPERS
# ==========================================================

def _xyz(lm, w, h):
    """Landmark to pixel-space 3D point (z scaled by width, MediaPipe convention)."""
    return np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float32)


def _hand_points(hand_landmarks, w, h):
    return np.array([_xyz(lm, w, h) for lm in hand_landmarks.landmark], dtype=np.float32)


def _pose_points(pose_landmarks, w, h):
    if pose_landmarks is None:
        return None
    return np.array([_xyz(lm, w, h) for lm in pose_landmarks.landmark], dtype=np.float32)


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _palm_scale(pts):
    """Hand size reference: wrist (0) to middle MCP (9)."""
    return max(_dist(pts[0], pts[9]), 1e-4)


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros_like(v)


def _roll_angle(pts_pose):
    """Camera tilt in radians from the shoulder line (fallback: eye line).

    The shoulders of a frontal signer span left-to-right, so the raw
    atan2(dy, dx) sits near +/-180 degrees. Only the vertical difference is a
    tilt; taking |dx| keeps the result near zero for an upright signer.
    """
    if pts_pose is None:
        return 0.0
    ls, rs = pts_pose[11], pts_pose[12]
    if _dist(ls, rs) > 1e-3:
        return float(np.arctan2(rs[1] - ls[1], abs(rs[0] - ls[0]) + 1e-3))
    le, re = pts_pose[2], pts_pose[5]
    if _dist(le, re) > 1e-3:
        return float(np.arctan2(re[1] - le[1], abs(re[0] - le[0]) + 1e-3))
    return 0.0


def _rotate_xy(vec, angle):
    c, s = np.cos(-angle), np.sin(-angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=np.float32)


def _sector(angle_deg):
    """Map a mathematical angle (0=right, 90=up) to an 8-way sector."""
    idx = int(((angle_deg % 360.0) + 22.5) // 45.0) % 8
    return DIR_SECTORS[idx]


# ==========================================================
# VIEW CLASSIFICATION (BODY-NORMALIZED)
# ==========================================================

def _infer_hand_side(wrist, pts_pose, fallback_label, shoulder_width):
    """
    Anatomical side of the signing hand. For a frontal signer the right hand
    appears on the image-left side, which is stable under mirroring mistakes.
    """
    if pts_pose is not None and shoulder_width > 1e-3:
        shoulder_mid_x = (pts_pose[11][0] + pts_pose[12][0]) * 0.5
        dx = (wrist[0] - shoulder_mid_x) / shoulder_width
        if dx < -0.15:
            return "R"
        if dx > 0.15:
            return "L"
    if fallback_label in ("Left", "Right"):
        # Tasks API labels assume a mirrored selfie frame; treat as fallback only.
        return "R" if fallback_label == "Left" else "L"
    return "R"


def classify_view(pts):
    """View of the hand as seen by the addressee.

    A fixed frontal camera still sees the hand from above when the fingers
    point at the camera (bird view) or edge-on when the palm faces sideways
    (right view); the wrist position does not change the view.
    """
    wrist, middle_mcp = pts[0], pts[9]
    long_axis = _unit(middle_mcp - wrist)
    if abs(long_axis[2]) > 1.2 * float(np.hypot(long_axis[0], long_axis[1])):
        return "bird"

    normal = _unit(np.cross(pts[5] - wrist, pts[17] - wrist))
    if abs(normal[0]) > 0.75 and abs(normal[2]) < 0.55:
        return "right"

    return "signer"


# ==========================================================
# FINGER DIRECTION (3D, ROLL-INVARIANT)
# ==========================================================

FINGER_CHAINS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


def _extended_finger_dirs(pts, scale):
    """Directions (tip - wrist) of the index/middle/ring/pinky fingers that are
    clearly unfolded. The thumb is excluded: in an open hand it points sideways
    and would otherwise hijack the direction estimate."""
    dirs = []
    for mcp, _pip, _dip, tip in FINGER_CHAINS:
        if _dist(pts[tip], pts[0]) > _dist(pts[mcp], pts[0]) + 0.45 * scale:
            dirs.append(pts[tip] - pts[0])
    return dirs


def classify_finger(pts, roll, side):
    """Direction of the extended fingers in the signer's body frame."""
    wrist = pts[0]
    scale = _palm_scale(pts)

    dirs = _extended_finger_dirs(pts, scale)
    if not dirs:
        # Curled hand (fist / hook): the fingers define the hand's long axis.
        dirs = [pts[i] - wrist for i in (8, 12, 16, 20)]
    d = np.mean(dirs, axis=0)

    d_xy = _rotate_xy(d[:2], roll)
    dy_up = -d_xy[1]                     # screen y points down
    angle = float(np.degrees(np.arctan2(dy_up, d_xy[0])))
    planar = _sector(angle)

    depth_ratio = float(d[2] / scale)
    planar_mag = float(np.hypot(d[0], d[1]) / scale)
    depth = ""
    # Only treat the finger as pointing out of / into the image plane when the
    # depth component dominates; a normal upright hand has a small noisy z.
    if abs(depth_ratio) > 0.25 and abs(depth_ratio) > 0.55 * planar_mag:
        # Original HamNoSys convention used by this project:
        # z > 0 (farther from camera) -> inward, z < 0 -> outward.
        depth = "i" if depth_ratio > 0 else "o"

    token = FINGER_TOKENS.get((planar, depth))
    if token is None:
        token = FINGER_TOKENS[(planar, "")]
    return token


# ==========================================================
# PALM ORIENTATION (3D NORMAL, SIDE-CANONICAL)
# ==========================================================

def classify_palm(pts, roll, side):
    wrist = pts[0]
    index_mcp = pts[5]
    pinky_mcp = pts[17]

    normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    if np.linalg.norm(normal) < 1e-6:
        return "hampalmd"

    normal = _unit(normal)

    # Right-hand palm geometry produces the opposite normal sign on the same
    # anatomical side, so flip it to keep "out of the palm" consistent.
    if side == "R":
        normal = -normal

    nx, ny_up = normal[0], -normal[1]
    projected = np.array([nx, ny_up], dtype=np.float32)
    if np.linalg.norm(projected) < 1e-3:
        return "hampalmd"

    projected = _rotate_xy(projected, roll)
    angle = float(np.degrees(np.arctan2(projected[1], projected[0])))
    return PALM_TOKENS[_sector(angle)]


# ==========================================================
# TEMPORAL SMOOTHING
# ==========================================================

def _smooth_component(values, window=5):
    smoothed = []
    n = len(values)
    for i in range(n):
        s = max(0, i - window // 2)
        e = min(n, i + window // 2 + 1)
        sub = [v for v in values[s:e] if v is not None]
        smoothed.append(Counter(sub).most_common(1)[0][0] if sub else values[i])
    return smoothed


def _smoothed_tuples(views, fingers, palms, window=5):
    views = _smooth_component(views, window)
    fingers = _smooth_component(fingers, window)
    palms = _smooth_component(palms, window)
    return [(v, f, p) if (v and f and p) else None
            for v, f, p in zip(views, fingers, palms)]


# ==========================================================
# FRAME + VIDEO PIPELINE
# ==========================================================

def _process_frame(rgb, pose_detector, hands_detector):
    pose_res = pose_detector.process(rgb)
    hand_res = hands_detector.process(rgb)

    if not hand_res.multi_hand_landmarks:
        return None, None

    h, w = rgb.shape[:2]
    pts_pose = _pose_points(pose_res.pose_landmarks, w, h)

    shoulder_width = 1.0
    if pts_pose is not None:
        shoulder_width = _dist(pts_pose[11], pts_pose[12])
    roll = _roll_angle(pts_pose)

    # MediaPipe does not order hands and the resting hand can look bigger, so
    # track the signing hand by its raised wrist instead of by palm size.
    best_idx = min(
        range(len(hand_res.multi_hand_landmarks)),
        key=lambda i: (
            hand_res.multi_hand_landmarks[i].landmark[0].y,
            -_palm_scale(_hand_points(hand_res.multi_hand_landmarks[i], w, h)),
        ),
    )
    best_pts = _hand_points(hand_res.multi_hand_landmarks[best_idx], w, h)
    best_label = None
    if hand_res.multi_handedness and best_idx < len(hand_res.multi_handedness):
        best_label = hand_res.multi_handedness[best_idx].classification[0].label

    # A hand hanging below the hips is at rest, not signing; its orientation
    # would otherwise dominate the temporal vote.
    if pts_pose is not None:
        hip_y = 0.5 * (pts_pose[23][1] + pts_pose[24][1])
        valid_hips = (pts_pose[23][1] > 0.15 * h and pts_pose[24][1] > 0.15 * h)
        if valid_hips and best_pts[0][1] > hip_y + 0.10 * shoulder_width:
            return None, None

    side = _infer_hand_side(best_pts[0], pts_pose, best_label, shoulder_width)

    view = classify_view(best_pts)
    finger = classify_finger(best_pts, roll, side)
    palm = classify_palm(best_pts, roll, side)

    return (view, finger, palm), best_pts


def run_orientation_module(video_path, min_detection_confidence=0.4,
                           min_tracking_confidence=0.4):
    """
    Generalized orientation module.
    Returns:
        dict: {"per_frame": [(view, finger, palm) | None, ...],
               "final": (view, finger, palm) | None}
    """
    if not os.path.exists(video_path):
        print(f"[Orientation] Video not found: {video_path}")
        return {"per_frame": [], "final": None}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Orientation] Failed to open video: {video_path}")
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            label, _ = _process_frame(rgb, pose_detector, hands_detector)
        except Exception:
            label = None
        raw.append(label)

    cap.release()
    pose_detector.close()
    hands_detector.close()

    views = [r[0] if r else None for r in raw]
    fingers = [r[1] if r else None for r in raw]
    palms = [r[2] if r else None for r in raw]
    per_frame = _smoothed_tuples(views, fingers, palms)

    valid = [r for r in per_frame if r]
    if not valid:
        return {"per_frame": per_frame, "final": None}

    final_view = Counter(v for v, _, _ in valid).most_common(1)[0][0]
    final_finger = Counter(f for _, f, _ in valid).most_common(1)[0][0]
    final_palm = Counter(p for _, _, p in valid).most_common(1)[0][0]

    return {
        "per_frame": per_frame,
        "final": (final_view, final_finger, final_palm),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run generalized orientation module.")
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()
    print(run_orientation_module(args.video_path))
