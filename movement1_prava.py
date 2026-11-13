#!/usr/bin/env python
# coding: utf-8
"""
Generalized & Optimized Movement 1 Classification Module for HamNoSys.

Key Generalizations:
1. Frame Rate (FPS) Invariance:
   - Velocities, accelerations, and jerk are now normalized by video FPS (dt = 1.0 / fps).
     Previously, 60 FPS videos were artificially categorized as 'hamslow' or 'hamnomotion'
     while 15 FPS videos were categorized as 'hamfast'.
2. Scale Invariance (Camera Distance / Zoom):
   - All spatial movements and amplitudes are normalized by the signer's hand scale
     (wrist-to-MCP distance). Speeds are calculated in 'hand-lengths per second',
     making classification invariant to close-up vs wide camera angles.
3. MediaPipe Z-Axis (Depth) Denoising:
   - MediaPipe depth (z) coordinates suffer from significant high-frequency noise.
     We apply trajectory smoothing (Exponential Moving Average) and calibrated depth weighting
     so depth jitter does not falsely trigger 'hammovei' / 'hammoveo'.
4. Encapsulation & Thread Safety:
   - Eliminated shared global `trajectory` deques that caused state leakage across video runs.
5. Robust Handling of Short Clips and Dropouts:
   - Handles variable-length trajectories down to 5 frames without crashing or returning None.
   - Returns consistent space-separated HamNoSys movement tokens.
"""

import os
import sys
import cv2
import numpy as np
from collections import deque, Counter

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


# ==========================================================
# GEOMETRIC & TRAJECTORY UTILITIES
# ==========================================================

def hand_center(landmarks):
    """Compute 3D centroid of hand landmarks."""
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark], dtype=np.float32)
    return pts.mean(axis=0)  # (x, y, z)


def get_hand_scale(landmarks):
    """Compute hand scale: distance between wrist (0) and middle MCP (9)."""
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark], dtype=np.float32)
    wrist = pts[0]
    middle_mcp = pts[9]
    scale = float(np.linalg.norm(wrist - middle_mcp))
    return max(scale, 1e-4)


def smooth_trajectory(traj, alpha=0.35):
    """
    Applies Exponential Moving Average (EMA) to smooth out camera jitter and landmark noise.
    Especially dampens Z-axis jitter.
    """
    if len(traj) < 2:
        return np.array(traj, dtype=np.float32)

    smoothed = [np.array(traj[0], dtype=np.float32)]
    for pt in traj[1:]:
        s = alpha * np.array(pt, dtype=np.float32) + (1.0 - alpha) * smoothed[-1]
        smoothed.append(s)

    return np.array(smoothed, dtype=np.float32)


# ==========================================================
# 1. DIRECTION CLASSIFICATION (FPS & SCALE INVARIANT)
# ==========================================================

def classify_direction(traj, scale=1.0, eps_per_sec=0.15, fps=30.0):
    """
    HamNoSys dominant instantaneous direction.
    Normalized by hand scale and frame time step dt = 1/fps.
    """
    if len(traj) < 3:
        return "hamnomotion"

    diffs = np.diff(traj, axis=0)  # shape (N-1, 3)
    # Velocity in hand-lengths per second
    vel = (diffs / scale) * fps

    dx, dy, dz = vel[:, 0], vel[:, 1], vel[:, 2]

    # Threshold for meaningful velocity
    eps = eps_per_sec

    # Z-axis requires slightly higher threshold due to MediaPipe depth noise
    eps_z = eps_per_sec * 1.5

    counts = {
        "u": int(np.sum(dy < -eps)),  # y decreases upwards in image coords
        "d": int(np.sum(dy >  eps)),  # y increases downwards
        "l": int(np.sum(dx < -eps)),  # x decreases leftwards
        "r": int(np.sum(dx >  eps)),  # x increases rightwards
        "i": int(np.sum(dz < -eps_z)), # depth towards camera
        "o": int(np.sum(dz >  eps_z)), # depth away from camera
    }

    total_valid = sum(counts.values())
    if total_valid < 3:
        return "hamnomotion"

    # Dominant per axis
    vy = max(counts["u"], counts["d"])
    vx = max(counts["l"], counts["r"])
    vz = max(counts["i"], counts["o"])

    vert = "u" if counts["u"] > counts["d"] else "d"
    hor  = "l" if counts["l"] > counts["r"] else "r"
    dep  = "i" if counts["i"] > counts["o"] else "o"

    # Primary single direction if strongly dominant
    if vy > 1.3 * max(vx, vz) and vy >= 3:
        return f"hammove{vert}"
    if vx > 1.3 * max(vy, vz) and vx >= 3:
        return f"hammove{hor}"
    if vz > 1.3 * max(vx, vy) and vz >= 3:
        return f"hammove{dep}"

    # Diagonal / combined directions
    if vy >= 2 and vx >= 2:
        return f"hammove{vert}{hor}"
    if vy >= 2 and vz >= 2:
        return f"hammove{vert}{dep}"
    if vx >= 2 and vz >= 2:
        return f"hammove{hor}{dep}"

    # Fallback to the single highest count
    best_axis = max([("u", counts["u"]), ("d", counts["d"]),
                     ("l", counts["l"]), ("r", counts["r"]),
                     ("i", counts["i"]), ("o", counts["o"])], key=lambda x: x[1])
    if best_axis[1] >= 2:
        return f"hammove{best_axis[0]}"

    return "hamnomotion"


# ==========================================================
# 2. SPEED CLASSIFICATION (SCALE & FPS INVARIANT)
# ==========================================================

def classify_speed(traj, scale=1.0, fps=30.0):
    """
    Speed in hand-lengths per second.
    Invariant to video framerate and camera zoom.
    """
    if len(traj) < 3:
        return None

    diffs = np.diff(traj, axis=0)
    # Speed in hand-lengths / second
    speeds_per_sec = (np.linalg.norm(diffs, axis=1) / scale) * fps
    mean_speed = float(np.mean(speeds_per_sec))

    if mean_speed > 1.8:
        return "hamfast"
    if mean_speed < 0.6:
        return "hamslow"
    return None


# ==========================================================
# 3. FORCE (TENSE VS REST)
# ==========================================================

def classify_force(traj, scale=1.0, fps=30.0):
    """
    Force based on normalized acceleration variance / jerk.
    """
    if len(traj) < 5:
        return "hamrest"

    accel = np.diff(np.diff(traj, axis=0), axis=0)
    jerk_per_sec = (np.linalg.norm(accel, axis=1) / scale) * (fps ** 2)
    mean_jerk = float(np.mean(jerk_per_sec))

    return "hamtense" if mean_jerk > 40.0 else "hamrest"


# ==========================================================
# 4. HALT (STOPPING AT END)
# ==========================================================

def detect_halt(traj, scale=1.0, fps=30.0):
    """
    Detects if movement decelerates to a stop at the end.
    """
    if len(traj) < 4:
        return None

    diffs = np.diff(traj[-4:], axis=0)
    end_speed = (np.linalg.norm(diffs, axis=1) / scale) * fps
    if end_speed[-1] < 0.2:
        return "hamhalt"
    return None


# ==========================================================
# 5. SIZE (AMPLITUDE)
# ==========================================================

def classify_size(traj, scale=1.0):
    """
    Movement amplitude measured in units of hand scale.
    """
    if len(traj) < 3:
        return None

    displacements = np.linalg.norm(traj - traj[0], axis=1) / scale
    max_amp = float(np.max(displacements))

    if max_amp < 0.4:
        return "hamsmallmod"
    if max_amp > 1.2:
        return "hamlargemod"
    return None


# ==========================================================
# 6. GROWTH
# ==========================================================

def classify_growth(traj, scale=1.0):
    """
    Evaluates whether distance from initial anchor expands or contracts.
    """
    if len(traj) < 6:
        return None

    d = np.linalg.norm(traj - traj[0], axis=1)
    d_start = np.mean(d[1:4])
    d_end   = np.mean(d[-3:])

    if d_end > d_start * 1.6 and d_end > 0.3 * scale:
        return "hamincreasing"
    if d_end < d_start * 0.6 and d_start > 0.3 * scale:
        return "hamdecreasing"
    return None


# ==========================================================
# 7. REPETITION & REVERSE
# ==========================================================

def classify_repetition(traj):
    """
    Detects oscillatory zero-crossings along primary axis of motion.
    """
    if len(traj) < 8:
        return None

    disp = traj - traj[0]
    # Find axis with largest variance
    axis = int(np.argmax(np.var(disp, axis=0)))
    dx = np.diff(traj[:, axis])

    # Zero crossings of velocity
    signs = np.sign(dx)
    zero_cross = int(np.sum(np.diff(signs) != 0))

    if zero_cross > 5:
        return "hamrepeatcontinueseveral"
    if zero_cross > 2:
        return "hamrepeatcontinue"
    return None


def classify_reverse(traj):
    """
    Detects direction reversal in dominant motion plane.
    """
    if len(traj) < 6:
        return None

    disp = traj - traj[0]
    axis = int(np.argmax(np.var(disp, axis=0)))
    dx = np.diff(traj[:, axis])
    signs = np.sign(dx)
    signs = signs[signs != 0]

    if len(signs) > 2 and np.any(signs[:-1] * signs[1:] < 0):
        return "hamrepeatreverse"
    return None


def classify_repeat_from_start(traj, scale=1.0):
    """
    Detects periodic returns to starting position.
    """
    if len(traj) < 10:
        return None

    start = traj[0]
    dists = np.linalg.norm(traj - start, axis=1) / scale

    # Points that leave and return near start (< 0.2 hand scale)
    near_start = dists < 0.25
    transitions = np.sum(np.diff(near_start.astype(int)) > 0)

    if transitions >= 3:
        return "hamrepeatfromstartseveral"
    if transitions >= 1:
        return "hamrepeatfromstart"
    return None


# ==========================================================
# 8. PATH SHAPE
# ==========================================================

def classify_path(traj):
    """
    Classifies 2D motion geometry (zigzag, wavy, ellipse).
    """
    if len(traj) < 8:
        return None

    diffs = np.diff(traj[:, :2], axis=0)
    magnitudes = np.linalg.norm(diffs, axis=1)
    valid_idx = magnitudes > 1e-4

    if np.sum(valid_idx) < 5:
        return None

    diffs_valid = diffs[valid_idx]
    angles = np.arctan2(diffs_valid[:, 1], diffs_valid[:, 0])
    angle_diff = np.abs(np.diff(angles))
    # Wrap to [-pi, pi]
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    angle_var = float(np.std(angle_diff))

    if angle_var > 1.2:
        return "hamzigzag"
    if 0.45 < angle_var <= 1.2:
        return "hamwavy"

    # Principal components / Covariance
    pts = traj[:, :2]
    pts = pts - np.mean(pts, axis=0)
    cov = np.cov(pts[:, 0], pts[:, 1])
    try:
        eigvals, _ = np.linalg.eig(cov)
        eigvals = np.sort(eigvals)[::-1]
        if eigvals[1] > 1e-6 and (eigvals[0] / eigvals[1]) > 2.5:
            if np.var(pts[:, 0]) > np.var(pts[:, 1]):
                return "hamellipseh"
            else:
                return "hamellipsev"
    except Exception:
        pass

    return None


# ==========================================================
# AGGREGATOR
# ==========================================================

def classify_movement1(trajectory_log):
    """
    Extracts dominant movement tokens from trajectory log.
    Returns clean, space-separated HamNoSys movement string.
    """
    if trajectory_log is None or len(trajectory_log) == 0:
        return "hamnomotion"

    keys = ["Direction", "Speed", "Force", "Halt", "Size", "Growth", "Repeat", "Reverse", "Path"]
    final_symbols = []

    for key in keys:
        vals = [x[key] for x in trajectory_log if key in x and x[key] is not None]
        if vals:
            dom = Counter(vals).most_common(1)[0][0]
            if dom not in ("none", "None", ""):
                final_symbols.append(dom)

    # Filter duplicates while maintaining sequence
    seen = set()
    unique_symbols = []
    for s in final_symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)

    return " ".join(unique_symbols) if unique_symbols else "hamnomotion"


# ==========================================================
# ENTRYPOINT
# ==========================================================

def run_movement1_module(video_path, min_detection_confidence=0.4, min_tracking_confidence=0.4):
    """
    Generalized Movement 1 Module.
    Works across videos of any framerate, resolution, scale, and duration.
    Returns:
        str: space-separated HamNoSys movement tokens.
    """
    if not os.path.exists(video_path):
        print(f"[Movement 1] Video not found: {video_path}")
        return "hamnomotion"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Movement 1] Failed to open video: {video_path}")
        return "hamnomotion"

    # Dynamically extract FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 5.0 or np.isnan(fps):
        fps = 30.0

    # Local trajectory deque (no global state leak)
    trajectory_points = deque(maxlen=int(fps * 2))  # ~2 seconds window
    scales = []
    label_log = []

    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)

        if res.multi_hand_landmarks:
            # Pick dominant hand by palm scale
            best_hand = None
            max_sc = -1.0
            for h in res.multi_hand_landmarks:
                sc = get_hand_scale(h)
                if sc > max_sc:
                    max_sc = sc
                    best_hand = h

            if best_hand is not None:
                center = hand_center(best_hand)
                scale = get_hand_scale(best_hand)
                trajectory_points.append(center)
                scales.append(scale)

                if len(trajectory_points) >= 5:
                    raw_traj = np.array(trajectory_points)
                    smooth_traj = smooth_trajectory(raw_traj)
                    avg_scale = float(np.median(scales[-len(trajectory_points):]))

                    direction = classify_direction(smooth_traj, scale=avg_scale, fps=fps)
                    speed     = classify_speed(smooth_traj, scale=avg_scale, fps=fps)
                    force     = classify_force(smooth_traj, scale=avg_scale, fps=fps)
                    halt      = detect_halt(smooth_traj, scale=avg_scale, fps=fps)
                    size      = classify_size(smooth_traj, scale=avg_scale)
                    growth    = classify_growth(smooth_traj, scale=avg_scale)
                    path      = classify_path(smooth_traj)
                    repeat_continue   = classify_repetition(smooth_traj)
                    repeat_from_start = classify_repeat_from_start(smooth_traj, scale=avg_scale)
                    repeat = repeat_from_start if repeat_from_start else repeat_continue
                    reverse = classify_reverse(smooth_traj)

                    label_log.append({
                        "Direction": direction,
                        "Speed": speed,
                        "Force": force,
                        "Halt": halt,
                        "Size": size,
                        "Growth": growth,
                        "Repeat": repeat,
                        "Reverse": reverse,
                        "Path": path
                    })

    cap.release()
    try:
        detector.close()
    except Exception:
        pass

    return classify_movement1(label_log)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run generalized Movement 1 module.")
    parser.add_argument("video_path", nargs="?", default="Prompt_1.mp4", help="Path to video file")
    args = parser.parse_args()

    out = run_movement1_module(args.video_path)
    print(f"Movement 1 Output: {out}")
