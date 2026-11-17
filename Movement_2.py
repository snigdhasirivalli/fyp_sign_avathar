#!/usr/bin/env python
# coding: utf-8
"""
Generalized Secondary-Movement (Movement 2) Module for HamNoSys.

Fixes for cross-video consistency:
1. Trajectories are divided by the signer's shoulder width and rotated by the
   shoulder roll, so pixels, resolution, zoom and camera tilt stop changing
   loop size and oscillation axes.
2. Velocity/oscillation counting is time-based (FPS-aware), not frame-count based.
3. Loop detection fits a circle to the drift-free trajectory and requires a
   large angular span, so waves and straight strokes are not called circles,
   and emits valid HamNoSys circle/clock directions (circleu/d/o/i/l/r,
   clocku/d/o/i/l/r) instead of plane names such as "hamclockxy".
4. Trajectory is EMA-smoothed and stored per video; no module-level deque leaks
   between videos.
5. One aligned label per frame, plus a final majority label.

Output:
    {"per_frame": [...], "trajectory": [...], "final": <label> | None}
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


def _xyz(lm, w, h):
    return np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float32)


def _pose_points(pose_landmarks, w, h):
    if pose_landmarks is None:
        return None
    return np.array([_xyz(lm, w, h) for lm in pose_landmarks.landmark], dtype=np.float32)


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _body_frame(pose_pts):
    """Return (shoulder_width, roll) for normalizing the trajectory."""
    if pose_pts is None:
        return 1.0, 0.0
    ls, rs = pose_pts[11], pose_pts[12]
    width = max(_dist(ls, rs), 1e-3)
    # The shoulder line spans left-to-right, so raw atan2(dy, dx) sits near
    # +/-180 deg; |dx| reduces it to the actual camera tilt.
    roll = float(np.arctan2(rs[1] - ls[1], abs(rs[0] - ls[0]) + 1e-3))
    return width, roll


def _to_body_frame(point, width, roll):
    """Pixel point -> shoulder-normalized, roll-corrected, math-oriented frame."""
    c, s = np.cos(-roll), np.sin(-roll)
    x = (point[0] * c - point[1] * s) / width
    y = (point[0] * s + point[1] * c) / width
    z = point[2] / width
    # Math frame: X right, Y up, Z towards the camera.
    return np.array([x, -y, -z], dtype=np.float32)


def _smooth_traj(traj, alpha=0.30):
    if len(traj) < 2:
        return np.array(traj, dtype=np.float32)
    out = [np.array(traj[0], dtype=np.float32)]
    for p in traj[1:]:
        out.append(alpha * np.array(p, dtype=np.float32) + (1 - alpha) * out[-1])
    return np.array(out, dtype=np.float32)


def _angular_momentum(traj):
    """Net rotation vector around the centroid plus its coherence."""
    c = traj.mean(axis=0)
    rel = traj - c
    moment = np.zeros(3, dtype=np.float32)
    abs_cross = 0.0
    for i in range(len(rel) - 1):
        cross = np.cross(rel[i], rel[i + 1])
        moment += cross
        abs_cross += float(np.linalg.norm(cross))
    coherence = float(np.linalg.norm(moment) / abs_cross) if abs_cross > 1e-9 else 0.0
    return moment, c, rel, coherence


def _axis_token(moment):
    idx = int(np.argmax(np.abs(moment)))
    value = float(moment[idx])
    if idx == 2:
        return "o" if value > 0 else "i"
    if idx == 1:
        return "u" if value > 0 else "d"
    return "r" if value > 0 else "l"


def _oscillation_label(traj, fps):
    if len(traj) < 6:
        return None

    dt = 1.0 / max(fps, 1.0)
    vel = np.diff(traj[:, :2], axis=0) / dt
    if len(vel) < 4:
        return None

    # Hysteresis: ignore tiny jitter when counting direction changes.
    eps = 0.05  # shoulder-widths per second

    def sign_changes(values):
        signs = []
        for v in values:
            if v > eps:
                signs.append(1)
            elif v < -eps:
                signs.append(-1)
        return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])

    v_changes = sign_changes(vel[:, 1])
    h_changes = sign_changes(vel[:, 0])

    var_y = float(np.var(traj[:, 1]))
    var_x = float(np.var(traj[:, 0]))
    duration = len(traj) * dt

    if duration < 0.35:
        return None

    if v_changes >= 3 and var_y > 1.3 * var_x:
        return "hamnodding"
    if h_changes >= 3 and var_x > 1.3 * var_y:
        return "hamswinging"
    return None


def _circle_fit(traj):
    """Least-squares circle through the x/y projection.

    Returns (radius, relative radial residual, angular span). A true loop
    covers a large angle around a small circle; a wave or a straight stroke
    either spans little angle or only fits a huge circle.
    """
    x, y = traj[:, 0], traj[:, 1]
    a = np.c_[2.0 * x, 2.0 * y, np.ones(len(x))]
    b = x * x + y * y
    (cx, cy, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    radius = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
    if radius < 1e-6:
        return 0.0, 1.0, 0.0
    residual = float(np.std(np.hypot(x - cx, y - cy) - radius) / radius)
    angles = np.unwrap(np.arctan2(y - cy, x - cx))
    span = float(angles.max() - angles.min())
    return radius, residual, span


def classify_movement2(traj, fps=30.0):
    """
    Classify a body-normalized 3D trajectory.
    Accepts either a list of points or a numpy array.
    """
    if traj is None or len(traj) < 6:
        return None

    traj = _smooth_traj(np.asarray(traj, dtype=np.float32))
    traj = traj - traj.mean(axis=0)
    # Remove linear drift (raising/lowering the hand, walking towards the
    # camera) so a loop that travels across the body still closes in the
    # residual motion instead of being mistaken for a straight movement.
    t = np.arange(len(traj), dtype=np.float32)
    for k in range(3):
        slope, intercept = np.polyfit(t, traj[:, k], 1)
        traj[:, k] -= slope * t + intercept

    plen = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))

    moment, _centroid, _rel, coherence = _angular_momentum(traj)
    fit_radius, fit_residual, span = _circle_fit(traj)

    # A loop must close on a hand-sized circle and sweep a large angle. A
    # back-and-forth wave covers almost no angle (or only fits a huge circle),
    # and a straight stroke is not circular at all.
    if (0.05 < fit_radius < 1.5 and span > 4.0 and fit_residual < 0.40
            and plen > 0.60):
        revs = span / (2.0 * np.pi)
        token = _axis_token(moment)
        # Body frame is math-oriented (Y up, Z toward camera): positive Z
        # angular momentum is counter-clockwise, i.e. a "circle" in HamNoSys.
        clockwise = moment[2] < 0

        if fit_radius < 0.35 and revs > 1.5:
            return "hamstircw" if clockwise else "hamstirccw"
        if revs > 1.5:
            return "hamclockfull" if clockwise else "hamcirclefull"
        return ("hamclock" if clockwise else "hamcircle") + token

    osc = _oscillation_label(traj, fps)
    if osc:
        return osc

    if plen < 0.10:
        return "unknown_movement2"
    return "unknown_movement2"


def _smooth_labels(values, window=7):
    smoothed = []
    n = len(values)
    for i in range(n):
        s = max(0, i - window // 2)
        e = min(n, i + window // 2 + 1)
        valid = [v for v in values[s:e] if v]
        smoothed.append(Counter(valid).most_common(1)[0][0] if valid else values[i])
    return smoothed


def run_movement2_module(video_path, min_detection_confidence=0.4,
                         min_tracking_confidence=0.4):
    """
    Generalized movement-2 module.
    Returns:
        dict: {"per_frame": [...], "trajectory": [...], "final": <label> | None}
    """
    if not os.path.exists(video_path):
        print(f"[Movement 2] Video not found: {video_path}")
        return {"per_frame": [], "trajectory": [], "final": None}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Movement 2] Failed to open video: {video_path}")
        return {"per_frame": [], "trajectory": [], "final": None}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 5 or np.isnan(fps):
        fps = 30.0

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

    window = max(int(fps * 2.0), 10)
    points = deque(maxlen=window)
    normalized = []
    raw_labels = []
    last_center = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_res = pose_detector.process(rgb)
        hand_res = hands_detector.process(rgb)
        pose_pts = _pose_points(pose_res.pose_landmarks, w, h)
        width, roll = _body_frame(pose_pts)

        if hand_res.multi_hand_landmarks:
            # Track the palm centre, not the wrist: waving pivots the hand at
            # the wrist, so a wrist path misses most of the actual movement.
            centers = []
            for hl in hand_res.multi_hand_landmarks:
                pts = np.array([_xyz(lm, w, h) for lm in hl.landmark],
                               dtype=np.float32)
                centers.append(pts[[0, 5, 9, 13, 17]].mean(axis=0))
            # Follow the hand temporally: with two hands the largest one can
            # change identity between frames and shred the trajectory.
            if last_center is not None:
                best = min(centers, key=lambda p: _dist(p, last_center))
                if _dist(best, last_center) > 0.9 * width:
                    # The tracked hand vanished and the resting hand was
                    # picked up instead. Start a new trajectory rather than
                    # connecting two different hands into a fake loop.
                    best = None
            else:
                best = min(centers, key=lambda p: p[1])
        else:
            best = None

        if best is not None:
            last_center = best
            body_pt = _to_body_frame(best, width, roll)
            points.append(body_pt)
            normalized.append(body_pt.tolist())
        else:
            last_center = None
            points.clear()
            normalized.append(None)

        if len(points) >= 6:
            raw_labels.append(classify_movement2(list(points), fps=fps))
        else:
            raw_labels.append(None)

    cap.release()
    pose_detector.close()
    hands_detector.close()

    per_frame = _smooth_labels(raw_labels)
    valid = [v for v in per_frame if v and v != "unknown_movement2"]
    final = Counter(valid).most_common(1)[0][0] if valid else None

    return {
        "per_frame": per_frame,
        "trajectory": normalized,
        "final": final,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run generalized movement-2 module.")
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()
    print(run_movement2_module(args.video_path))
