#!/usr/bin/env python
# coding: utf-8
"""
Test Runner for Generalized HamNoSys Modules:
- Handshape_Model.py
- fing_locations_d.py
- movement1_prava.py

Scans for available videos in the directory and the parent Integration folder,
runs all 3 generalized modules, and displays comparison outputs.
"""

import os
import sys
import json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_DIR = os.path.join(os.path.dirname(THIS_DIR), "Integration")

# Import the 3 generalized modules locally
sys.path.insert(0, THIS_DIR)
from Handshape_Model import run_handshape_module
from fing_locations_d import run_finger_location_module
from movement1_prava import run_movement1_module

def find_video_files():
    """Find mp4 videos in either this folder or Integration folder."""
    candidates = []
    for d in [THIS_DIR, INTEGRATION_DIR]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(".mp4") and not f.endswith("_annotated.mp4"):
                    full_p = os.path.join(d, f)
                    if full_p not in candidates:
                        candidates.append(full_p)
    return candidates

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

        try:
            handshape_out = run_handshape_module(video_path)
            handshape_label = handshape_out.get("final") if isinstance(handshape_out, dict) else str(handshape_out)
            handshape_frames = len(handshape_out.get("per_frame", [])) if isinstance(handshape_out, dict) else 0
        except Exception as e:
            handshape_label = f"ERROR: {e}"
            handshape_frames = 0

        try:
            finger_out = run_finger_location_module(video_path)
            finger_label = finger_out.get("final") if isinstance(finger_out, dict) else str(finger_out)
            finger_frames = len(finger_out.get("per_frame", [])) if isinstance(finger_out, dict) else 0
        except Exception as e:
            finger_label = f"ERROR: {e}"
            finger_frames = 0

        try:
            movement_out = run_movement1_module(video_path)
        except Exception as e:
            movement_out = f"ERROR: {e}"

        print(f"  -> Handshape (Frames: {handshape_frames}): {handshape_label}")
        print(f"  -> Finger Location (Frames: {finger_frames}): {finger_label}")
        print(f"  -> Movement 1: {movement_out}")

        results.append({
            "video": vname,
            "handshape": handshape_label,
            "finger_location": finger_label,
            "movement1": movement_out
        })

    out_file = os.path.join(THIS_DIR, "generalized_test_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Testing complete! Results saved to: {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
