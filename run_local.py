"""
Local runner for the HamNoSys Integration Pipeline.
Adapted from the Colab notebook to run on Windows without Google Drive.
"""

import sys
import os

# ── Mock google.colab so notebook-converted modules import without error ──
import types as _types

_google_mod   = _types.ModuleType('google')
_colab_mod    = _types.ModuleType('google.colab')
_drive_mod    = _types.ModuleType('google.colab.drive')
_patches_mod  = _types.ModuleType('google.colab.patches')
_files_mod    = _types.ModuleType('google.colab.files')

_drive_mod.mount          = lambda *a, **kw: None
_patches_mod.cv2_imshow   = lambda img: None          # no-op display
_files_mod.download       = lambda *a, **kw: None

_colab_mod.drive   = _drive_mod
_colab_mod.patches = _patches_mod
_colab_mod.files   = _files_mod

sys.modules.setdefault('google', _google_mod)
sys.modules['google.colab']         = _colab_mod
sys.modules['google.colab.drive']   = _drive_mod
sys.modules['google.colab.patches'] = _patches_mod
sys.modules['google.colab.files']   = _files_mod

# ── Compatibility shim: patches mp.solutions for mediapipe >= 0.10.30 ──
# Must be imported BEFORE any module that uses mp.solutions.hands / mp.solutions.pose
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mediapipe_solutions_compat  # noqa: F401  (side-effect import)

# Add current directory to path so all modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import mediapipe as mp

# Import all sub-modules
from Handshape_Model import run_handshape_module
from ori_model2 import run_orientation_module
from upper_body_locations_video import run_upper_body_location_module
from Head_and_face_location import run_head_face_location_module
from hand_location_video_P import run_hand_location_module
from fing_locations_d import run_finger_location_module
from contact_types_e import run_contact_type_module
from Arm_and_Space_positions import run_arm_space_module
from movement1_prava import run_movement1_module
from Movement_2 import run_movement2_module
from two_hands import run_two_hands_module

# =============================
# VIDEO PATH (local)
# =============================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(THIS_DIR, "Prompt_1.mp4")
output_video = os.path.join(THIS_DIR, "output_local.mp4")

if not os.path.exists(video_path):
    print(f"[ERROR] Video not found: {video_path}")
    print("Available videos:")
    for f in os.listdir(THIS_DIR):
        if f.endswith(".mp4"):
            print(f"  - {f}")
    sys.exit(1)

print(f"[INFO] Using video: {video_path}")

# =============================
# RUN ALL MODULES
# =============================
print("\nRunning modules (this may take a minute)...")

handshape       = run_handshape_module(video_path)
orientation     = run_orientation_module(video_path)
upper_body      = run_upper_body_location_module(video_path)
head_face       = run_head_face_location_module(video_path)
hand_location   = run_hand_location_module(video_path)
finger_location = run_finger_location_module(video_path)
contact         = run_contact_type_module(video_path)
arm_space       = run_arm_space_module(video_path)
movement1       = run_movement1_module(video_path)
movement2       = run_movement2_module(video_path)
two_hands       = run_two_hands_module(video_path)

# =============================
# EXTRACT LABELS
# =============================
def extract_labels(module_output):
    if isinstance(module_output, dict):
        if "per_frame" in module_output:
            labels = module_output["per_frame"]
            cleaned = []
            for item in labels:
                if item is None:
                    continue
                if isinstance(item, tuple):
                    cleaned.extend([str(x) for x in item if x is not None])
                elif isinstance(item, dict):
                    cleaned.extend(v for v in item.values()
                                   if isinstance(v, str) and v.startswith("ham"))
                else:
                    cleaned.append(str(item))
            return cleaned
    if isinstance(module_output, str):
        return [module_output]
    return []


def combine_hamnosys(*modules):
    final = []
    for m in modules:
        final.extend(extract_labels(m))
    seen = set()
    clean = []
    for x in final:
        if x not in seen:
            clean.append(x)
            seen.add(x)
    return " ".join(clean)


# =============================
# GENERATE HAMNOSYS OUTPUT
# =============================
def generate_hamnosys():
    print("\n===== MODULE OUTPUTS =====")
    print("Handshape       :", handshape)
    print("Orientation     :", orientation)
    print("Arm & Space     :", arm_space)
    print("Upper Body      :", upper_body)
    print("Head & Face     :", head_face)
    print("Hand Location   :", hand_location)
    print("Finger Location :", finger_location)
    print("Contact Type    :", contact)
    print("Movement 1      :", movement1)
    print("Movement 2      :", movement2)
    print("Two Hands       :", two_hands)

    hamnosys_code = combine_hamnosys(
        handshape, orientation, upper_body, head_face,
        hand_location, finger_location, contact, arm_space,
        movement1, movement2, two_hands
    )

    print("\n========== FINAL HAMNOSYS ==========")
    print(hamnosys_code)
    return hamnosys_code


hamnosys_output = generate_hamnosys()

# =============================
# CREATE OUTPUT VIDEO WITH LABELS
# =============================
def safe(x):
    if x is None:
        return "none"
    return str(x)

def get_frames(output, total_frames):
    if isinstance(output, dict):
        arr = output.get("per_frame", [])
        if len(arr) > 0:
            return arr
    if isinstance(output, str):
        return [output] * total_frames
    if isinstance(output, list):
        return output
    return ["none"] * total_frames


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {video_path}")
    sys.exit(1)

ret, frame = cap.read()
h, w, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 25, (w, h))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"\n[INFO] Writing annotated output video ({total_frames} frames)...")

handshape_frames    = get_frames(handshape, total_frames)
orientation_frames  = get_frames(orientation, total_frames)
upper_frames        = get_frames(upper_body, total_frames)
head_frames         = get_frames(head_face, total_frames)
hand_frames         = get_frames(hand_location, total_frames)
finger_frames       = get_frames(finger_location, total_frames)
contact_frames      = get_frames(contact, total_frames)
arm_frames          = get_frames(arm_space, total_frames)
movement1_frames    = get_frames(movement1, total_frames)
movement2_frames    = get_frames(movement2, total_frames)

raw_two_frames = two_hands.get("per_frame", []) if isinstance(two_hands, dict) else []
two_hands_frames = [
    " ".join(v for v in f.values() if isinstance(v, str) and v.startswith("ham"))
    if isinstance(f, dict) else safe(f)
    for f in raw_two_frames
]
if not two_hands_frames:
    two_hands_frames = get_frames(two_hands, total_frames)

frame_id = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    def safe_get(arr, idx):
        try:
            return safe(arr[idx])
        except IndexError:
            return "none"

    handshape_label    = safe_get(handshape_frames, frame_id)
    orientation_label  = safe_get(orientation_frames, frame_id)
    location_label = " ".join([
        safe_get(upper_frames, frame_id),
        safe_get(head_frames, frame_id),
        safe_get(hand_frames, frame_id),
        safe_get(finger_frames, frame_id),
        safe_get(contact_frames, frame_id),
        safe_get(arm_frames, frame_id),
    ])
    movement_label = " ".join([
        safe_get(movement1_frames, frame_id),
        safe_get(movement2_frames, frame_id),
    ])
    two_hands_label = safe_get(two_hands_frames, frame_id)

    # Draw labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Handshape: {handshape_label}",   (30, 40),  font, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"Orientation: {orientation_label}",(30, 70),  font, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"Location: {location_label}",      (30, 110), font, 0.5, (0,255,255), 2)
    cv2.putText(frame, f"Movement: {movement_label}",      (30, 150), font, 0.6, (255,255,0), 2)
    cv2.putText(frame, f"TwoHands: {two_hands_label}",     (30, 190), font, 0.6, (255,0,255), 2)

    out.write(frame)
    frame_id += 1

cap.release()
out.release()

print(f"[DONE] Output video saved to: {output_video}")
print(f"[DONE] HamNoSys: {hamnosys_output}")
