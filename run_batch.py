"""
Batch HamNoSys pipeline runner.
Processes 4 videos and saves outputs (text + annotated video)
into a hamnosys_outputs/ folder inside the Integration directory.
"""

import sys
import os
import json
import types as _types

# ── Mock google.colab ──
_google_mod   = _types.ModuleType('google')
_colab_mod    = _types.ModuleType('google.colab')
_drive_mod    = _types.ModuleType('google.colab.drive')
_patches_mod  = _types.ModuleType('google.colab.patches')
_files_mod    = _types.ModuleType('google.colab.files')
_drive_mod.mount        = lambda *a, **kw: None
_patches_mod.cv2_imshow = lambda img: None
_files_mod.download     = lambda *a, **kw: None
_colab_mod.drive   = _drive_mod
_colab_mod.patches = _patches_mod
_colab_mod.files   = _files_mod
sys.modules.setdefault('google', _google_mod)
sys.modules['google.colab']         = _colab_mod
sys.modules['google.colab.drive']   = _drive_mod
sys.modules['google.colab.patches'] = _patches_mod
sys.modules['google.colab.files']   = _files_mod

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import mediapipe_solutions_compat  # noqa: F401

import numpy as np
import cv2

# ── Import all sub-modules ──
from Handshape_Model            import run_handshape_module
from ori_model2                 import run_orientation_module
from upper_body_locations_video import run_upper_body_location_module
from Head_and_face_location     import run_head_face_location_module
from hand_location_video_P      import run_hand_location_module
from fing_locations_d           import run_finger_location_module
from contact_types_e            import run_contact_type_module
from Arm_and_Space_positions    import run_arm_space_module
from movement1_prava            import run_movement1_module
from Movement_2                 import run_movement2_module
from two_hands                  import run_two_hands_module

# ── Output folder ──
OUTPUT_DIR = os.path.join(THIS_DIR, "outputt_2")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n[INFO] Output folder: {OUTPUT_DIR}\n")

# ── Input videos ──
VIDEOS = [
    "Prompt_1.mp4",
]

# ── Helpers ──
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


def safe_get(arr, idx):
    try:
        return safe(arr[idx])
    except (IndexError, TypeError):
        return "none"


def get_final(output):
    """Get the dominant/final label from a module output, cleanly formatted."""
    if isinstance(output, dict):
        # Use explicit 'final' key if available
        val = output.get("final", None)
        if val is None:
            # Fall back to most common label in per_frame
            frames = output.get("per_frame", [])
            non_none = [str(x) for x in frames if x is not None and str(x) not in ("none", "None", "")]
            if non_none:
                from collections import Counter
                val = Counter(non_none).most_common(1)[0][0]
            else:
                return "none"
        # Clean up tuple / numpy repr  e.g. (np.str_('signer'), ...)
        if isinstance(val, tuple):
            return " ".join(_clean(v) for v in val)
        return _clean(str(val))
    if isinstance(output, str):
        return output
    return "none"


def _clean(s):
    """Strip numpy wrapper strings like np.str_('value') -> value."""
    import re
    s = str(s)
    # Handle np.str_('value') or np.str_("value")
    m = re.match(r"np\.str_\(['\"](.+?)['\"]\)", s)
    if m:
        return m.group(1)
    return s


def count_tokens(output, video_frames=None):
    """Count total frames and tokens from a module output.

    Tuples are expanded (e.g. orientation frames have 3-element tuples).
    'none' and None are INCLUDED in the token count.
    Returns a dict with:
      - 'total_frames': int (number of frames processed)
      - 'total_tokens': int (total token count across all frames)
      - 'unique_tokens': int (count of unique tokens)
      - 'tokens': sorted list of unique tokens
      - 'token_counts': dictionary of token -> frequency count
    """
    from collections import Counter
    if isinstance(output, str):
        # Movement1 is a space-separated token sequence
        tokens = [t for t in output.split() if t]
        return {
            "total_frames": video_frames if video_frames is not None else 1,
            "total_tokens": len(tokens),
            "unique_tokens": len(set(tokens)),
            "tokens": sorted(set(tokens)),
            "token_counts": dict(Counter(tokens)),
        }
    if isinstance(output, dict):
        frames = output.get("per_frame", [])
        if not frames and "trajectory" in output:
            traj = output.get("trajectory", [])
            final_val = output.get("final")
            tokens = []
            if final_val is not None:
                tokens.append(_clean(str(final_val)))
            return {
                "total_frames": len(traj),
                "total_tokens": len(tokens),
                "unique_tokens": len(set(tokens)),
                "tokens": sorted(set(tokens)),
                "token_counts": dict(Counter(tokens)),
            }

        flat = []
        for item in frames:
            if item is None:
                flat.append("none")
            elif isinstance(item, tuple):
                for x in item:
                    if x is None:
                        flat.append("none")
                    else:
                        val = _clean(str(x))
                        flat.append("none" if val.lower() == "none" else val)
            elif isinstance(item, dict):
                tok = [v for v in item.values()
                       if isinstance(v, str) and v.startswith("ham")]
                flat.extend(tok if tok else ["none"])
            else:
                val = _clean(str(item))
                flat.append("none" if val.lower() == "none" else val)
        return {
            "total_frames": len(frames),
            "total_tokens": len(flat),
            "unique_tokens": len(set(flat)),
            "tokens": sorted(set(flat)),
            "token_counts": dict(Counter(flat)),
        }
    return {
        "total_frames": 0,
        "total_tokens": 0,
        "unique_tokens": 0,
        "tokens": [],
        "token_counts": {},
    }


def _clean_obj(obj):
    if isinstance(obj, dict):
        return {k: _clean_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_obj(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_clean_obj(x) for x in obj)
    elif hasattr(obj, "item"):
        return obj.item()
    elif isinstance(obj, str):
        return _clean(obj)
    return obj


def format_verbose_output(output):
    """Return a string representation of a module output showing per_frame.

    Mirrors the format shown in the user example:
      {'per_frame': [...], 'final': ...}
    For movement2 we also include the trajectory key.
    """
    if isinstance(output, str):
        return output
    cleaned = _clean_obj(output)
    return repr(cleaned)


# ── Main batch loop ──
# Load existing summary if present (so we append, not overwrite)
summary_path = os.path.join(OUTPUT_DIR, "summary.json")
if os.path.exists(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    print(f"[INFO] Loaded {len(summary)} existing results from summary.json\n")
else:
    summary = []

for video_name in VIDEOS:
    video_path = os.path.join(THIS_DIR, video_name)
    stem = os.path.splitext(video_name)[0].replace(" ", "_")

    print("=" * 60)
    print(f"[VIDEO] {video_name}")
    print("=" * 60)

    if not os.path.exists(video_path):
        print(f"  [SKIP] File not found: {video_path}")
        continue

    # Get total video frames for reference
    cap_temp = cv2.VideoCapture(video_path)
    video_total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_temp.isOpened() else None
    cap_temp.release()

    print("  Running modules...")
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

    hamnosys_code = combine_hamnosys(
        handshape, orientation, upper_body, head_face,
        hand_location, finger_location, contact, arm_space,
        movement1, movement2, two_hands
    )

    # ── Compute per-module token counts ──
    modules_info = [
        ("Handshape",       handshape),
        ("Orientation",     orientation),
        ("Arm & Space",     arm_space),
        ("Upper Body",      upper_body),
        ("Head & Face",     head_face),
        ("Hand Location",   hand_location),
        ("Finger Location", finger_location),
        ("Contact Type",    contact),
        ("Movement 1",      movement1),
        ("Movement 2",      movement2),
        ("Two Hands",       two_hands),
    ]
    token_counts = {name: count_tokens(out, video_frames=video_total_frames) for name, out in modules_info}

    # ── Console output ──
    print(f"\n===== MODULE OUTPUTS =====")
    print(f"Handshape       : {format_verbose_output(handshape)}")
    print(f"Orientation     : {format_verbose_output(orientation)}")
    print(f"Arm & Space     : {format_verbose_output(arm_space)}")
    print(f"Upper Body      : {format_verbose_output(upper_body)}")
    print(f"Head & Face     : {format_verbose_output(head_face)}")
    print(f"Hand Location   : {format_verbose_output(hand_location)}")
    print(f"Finger Location : {format_verbose_output(finger_location)}")
    print(f"Contact Type    : {format_verbose_output(contact)}")
    print(f"Movement 1      : {movement1 if isinstance(movement1, str) else format_verbose_output(movement1)}")
    print(f"Movement 2      : {format_verbose_output(movement2)}")
    print(f"Two Hands       : {format_verbose_output(two_hands)}")
    print(f"\n========== FINAL HAMNOSYS ==========")
    print(hamnosys_code)

    print(f"\n===== MODULE STATS (TOTAL FRAMES & TOTAL TOKENS) =====")
    for mod_name, tc in token_counts.items():
        print(f"  {mod_name:<20}: {tc['total_frames']} frames | {tc['total_tokens']} total tokens ({tc['unique_tokens']} unique) -> {tc['tokens']}")

    # ── Save HamNoSys text file ──
    txt_path = os.path.join(OUTPUT_DIR, f"{stem}_hamnosys.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Video: {video_name}\n")
        f.write("=" * 50 + "\n\n")

        f.write("===== MODULE OUTPUTS =====\n")
        f.write(f"Handshape       : {format_verbose_output(handshape)}\n")
        f.write(f"Orientation     : {format_verbose_output(orientation)}\n")
        f.write(f"Arm & Space     : {format_verbose_output(arm_space)}\n")
        f.write(f"Upper Body      : {format_verbose_output(upper_body)}\n")
        f.write(f"Head & Face     : {format_verbose_output(head_face)}\n")
        f.write(f"Hand Location   : {format_verbose_output(hand_location)}\n")
        f.write(f"Finger Location : {format_verbose_output(finger_location)}\n")
        f.write(f"Contact Type    : {format_verbose_output(contact)}\n")
        f.write(f"Movement 1      : {movement1 if isinstance(movement1, str) else format_verbose_output(movement1)}\n")
        f.write(f"Movement 2      : {format_verbose_output(movement2)}\n")
        f.write(f"Two Hands       : {format_verbose_output(two_hands)}\n")

        f.write("\n========== FINAL HAMNOSYS ==========\n")
        f.write(hamnosys_code + "\n")

        f.write("\n===== MODULE STATS (TOTAL FRAMES & TOTAL TOKENS) =====\n")
        for mod_name, tc in token_counts.items():
            f.write(f"  {mod_name:<20}:\n")
            f.write(f"    Total Frames      : {tc['total_frames']}\n")
            f.write(f"    Total Tokens      : {tc['total_tokens']}\n")
            f.write(f"    Unique Tokens     : {tc['unique_tokens']}\n")
            f.write(f"    Tokens            : {tc['tokens']}\n")
            f.write(f"    Frequency Counts  : {tc['token_counts']}\n")

    print(f"\n  [SAVED] Text  -> {txt_path}")

    # ── Create annotated output video ──
    out_video_path = os.path.join(OUTPUT_DIR, f"{stem}_annotated.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open video for annotation: {video_path}")
    else:
        ret, first_frame = cap.read()
        if ret:
            h, w, _ = first_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_video_path, fourcc, 25, (w, h))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"  [INFO] Annotating {total_frames} frames...")

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

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_id = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                location_label = " | ".join([
                    safe_get(upper_frames, frame_id),
                    safe_get(hand_frames, frame_id),
                    safe_get(finger_frames, frame_id),
                    safe_get(contact_frames, frame_id),
                    safe_get(arm_frames, frame_id),
                ])
                movement_label = " | ".join([
                    safe_get(movement1_frames, frame_id),
                    safe_get(movement2_frames, frame_id),
                ])

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"Handshape:   {safe_get(handshape_frames, frame_id)}",   (20, 35),  font, 0.55, (0, 255, 0),   2)
                cv2.putText(frame, f"Orientation: {safe_get(orientation_frames, frame_id)}", (20, 65),  font, 0.55, (0, 255, 0),   2)
                cv2.putText(frame, f"Location:    {location_label}",                          (20, 100), font, 0.45, (0, 255, 255), 2)
                cv2.putText(frame, f"Movement:    {movement_label}",                          (20, 135), font, 0.45, (255, 255, 0), 2)

                display_code = hamnosys_code[:80] + ("..." if len(hamnosys_code) > 80 else "")
                cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
                cv2.putText(frame, display_code, (10, h - 12), font, 0.4, (255, 255, 255), 1)

                out.write(frame)
                frame_id += 1

            cap.release()
            out.release()
            print(f"  [SAVED] Video -> {out_video_path}")
        else:
            cap.release()
            print(f"  [WARN] Could not read first frame from {video_path}")

    entry = {
        "video": video_name,
        "hamnosys": hamnosys_code,
        "txt_file": txt_path,
        "annotated_video": out_video_path,
    }
    existing_idx = next((i for i, item in enumerate(summary) if item.get("video") == video_name), None)
    if existing_idx is not None:
        summary[existing_idx] = entry
    else:
        summary.append(entry)
    print()

# ── Save updated summary JSON ──
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("ALL DONE — HAMNOSYS SUMMARY")
print("=" * 60)
for entry in summary:
    print(f"\n  Video   : {entry['video']}")
    print(f"  HamNoSys: {entry['hamnosys']}")
print(f"\n  Output folder : {OUTPUT_DIR}")
print(f"  Summary JSON  : {summary_path}")
