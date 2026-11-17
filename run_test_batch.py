#!/usr/bin/env python
# coding: utf-8
"""
Test Runner for all Generalized HamNoSys Modules.

Runs every generalized module on the videos found in this directory or the
parent Integration folder and writes the per-video dominant labels to
generalized_test_results.json.
"""

import os
import sys
import json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_DIR = os.path.join(os.path.dirname(THIS_DIR), "Integration")

sys.path.insert(0, THIS_DIR)

from Handshape_Model import run_handshape_module
from fing_locations_d import run_finger_location_module
from movement1_prava import run_movement1_module
from ori_model2 import run_orientation_module
from upper_body_locations_video import run_upper_body_location_module
from Head_and_face_location import run_head_face_location_module
from hand_location_video_P import run_hand_location_module
from contact_types_e import run_contact_type_module
from Arm_and_Space_positions import run_arm_space_module
from Movement_2 import run_movement2_module


MODULES = [
    ("handshape", run_handshape_module),
    ("finger_location", run_finger_location_module),
    ("movement1", run_movement1_module),
    ("orientation", run_orientation_module),
    ("upper_body", run_upper_body_location_module),
    ("head_face", run_head_face_location_module),
    ("hand_location", run_hand_location_module),
    ("contact", run_contact_type_module),
    ("arm_space", run_arm_space_module),
    ("movement2", run_movement2_module),
]


def find_video_files():
    """Find mp4 videos in either this folder or the Integration folder."""
    candidates = []
    for d in [THIS_DIR, INTEGRATION_DIR]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(".mp4") and not f.endswith("_annotated.mp4"):
                    full_p = os.path.join(d, f)
                    if full_p not in candidates:
                        candidates.append(full_p)
    return candidates


def summarize(output):
    if isinstance(output, dict):
        final = output.get("final")
        frames = len(output.get("per_frame", []))
        return final, frames
    return output, 0


def main():
    videos = find_video_files()
    if not videos:
        print("[WARN] No .mp4 videos found to test.")
        return

    print("=" * 70)
    print("TESTING GENERALIZED & OPTIMIZED MODULES ACROSS VIDEOS")
    print("=" * 70)
    print(f"Found {len(videos)} videos:\n" + "\n".join(f"  - {os.path.basename(v)}" for v in videos))
    print("=" * 70)

    results = []

    for video_path in videos:
        vname = os.path.basename(video_path)
        print(f"\n[PROCESSING] {vname} ...")
        row = {"video": vname}

        for name, fn in MODULES:
            try:
                label, frames = summarize(fn(video_path))
                row[name] = label
                print(f"  -> {name:16s} (Frames: {frames}): {label}")
            except Exception as e:
                row[name] = f"ERROR: {e}"
                print(f"  -> {name:16s} ERROR: {e}")

        results.append(row)

    out_file = os.path.join(THIS_DIR, "generalized_test_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Testing complete! Results saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
