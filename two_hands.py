#!/usr/bin/env python
# coding: utf-8
"""
Two-Handed Signs Module — HamNoSys Pipeline
============================================
Detects and classifies ALL two-handed relationship signs from the
HamNoSys "Two-handed" keyboard tab:

ROW 1  — Simultaneity / Sequencing:
  hamparallel  hamsequential  hamseqleftright  hamseqatloc  hamseqoverlap

ROW 2  — Relationship / Symmetry brackets:
  hamsymmlr   hamsymmr   hamsymmrl   hamcircular

ROW 3  — Dominant / Non-dominant hand:
  hamreplace   hamsimultaneous

ROW 4  — Secondary hand articulator (hold):
  hamholdsecond

ROW 5  — Secondary hand shape brackets (passive hand):
  hamsecleftpar  hamsecpar  hamsecleftbrack  hamsecbrack
  hamsecleftangle  hamsecangle  hamsecleftbrace  hamsecbrace  hamsecbar

ROW 6  — Passive hand shape presets:
  hamsecshape0  hamsecshape1  hamsecshape2  hamsecshape3
  hamsecshape4  hamsecshape5  hamsecshape6

ROW 7  — Secondary body location qualifiers:
  hamsecloc_comma  hamsecloc_dot  hamsecloc_bar  hamsecloc_q  hamsecloc_cup

Integration connections:
  imports shared_landmarks.get_video_landmarks()
  run_two_hands_module(video_path) -> dict consumed by integration_pipeline
  inject_two_hands(sequence_list, result) patches combined HamNoSys sequence
"""

import os
import sys
import cv2
import numpy as np
from collections import Counter, deque

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

try:
    from shared_landmarks import get_video_landmarks
    _USE_SHARED = True
except ImportError:
    _USE_SHARED = False

import mediapipe as mp

# The repository pins modern MediaPipe, where the legacy mp.solutions API is
# provided by this compatibility layer.
try:
    import mediapipe_solutions_compat  # noqa: F401
except ImportError:
    pass

_mp_hands = mp.solutions.hands

# ─── HamNoSys TWO-HANDED TOKEN CATALOGUE ────────────────────────────────────

TWO_HAND_SEQ_TOKENS = [
    "hamparallel",
    "hamsequential",
    "hamseqleftright",
    "hamseqatloc",
    "hamseqoverlap",
]

TWO_HAND_SYMM_TOKENS = [
    "hamsymmlr",
    "hamsymmr",
    "hamsymmrl",
    "hamcircular",
]

TWO_HAND_ROLE_TOKENS = [
    "hamreplace",
    "hamsimultaneous",
]

TWO_HAND_HOLD_TOKENS = [
    "hamholdsecond",
]

TWO_HAND_SEC_BRACKET_TOKENS = [
    "hamsecleftpar",
    "hamsecpar",
    "hamsecleftbrack",
    "hamsecbrack",
    "hamsecleftangle",
    "hamsecangle",
    "hamsecleftbrace",
    "hamsecbrace",
    "hamsecbar",
]

TWO_HAND_SHAPE_TOKENS = [
    "hamsecshape0",
    "hamsecshape1",
    "hamsecshape2",
    "hamsecshape3",
    "hamsecshape4",
    "hamsecshape5",
    "hamsecshape6",
]

TWO_HAND_LOC_TOKENS = [
    "hamsecloc_comma",
    "hamsecloc_dot",
    "hamsecloc_bar",
    "hamsecloc_q",
    "hamsecloc_cup",
]

ALL_TWO_HAND_TOKENS = (
    TWO_HAND_SEQ_TOKENS +
    TWO_HAND_SYMM_TOKENS +
    TWO_HAND_ROLE_TOKENS +
    TWO_HAND_HOLD_TOKENS +
    TWO_HAND_SEC_BRACKET_TOKENS +
    TWO_HAND_SHAPE_TOKENS +
    TWO_HAND_LOC_TOKENS
)

# ─── GEOMETRIC HELPERS ───────────────────────────────────────────────────────

def _lm_to_arr(lm_obj):
    if lm_obj is None:
        return None
    try:
        pts = [(lm.x, lm.y, lm.z) for lm in lm_obj.landmark]
    except AttributeError:
        return None
    return np.array(pts, dtype=np.float32)


def _wrist(arr):
    return arr[0] if arr is not None else None


def _palm_center(arr):
    if arr is None:
        return None
    return np.mean(arr[[0, 5, 9, 13, 17]], axis=0)


def _dist(a, b):
    if a is None or b is None:
        return 9999.0
    return float(np.linalg.norm(a - b))


def _finger_states(arr):
    CHAINS = {
        "thumb":  (1, 2, 3, 4),
        "index":  (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring":   (13, 14, 15, 16),
        "pinky":  (17, 18, 19, 20),
    }
    if arr is None:
        return {}
    states = {}
    for name, (mcp, pip, dip, tip) in CHAINS.items():
        direct = _dist(arr[mcp], arr[tip])
        chain  = (_dist(arr[mcp], arr[pip]) +
                  _dist(arr[pip], arr[dip]) +
                  _dist(arr[dip], arr[tip]))
        ratio  = direct / (chain + 1e-8)
        if ratio >= 0.76:
            states[name] = "straight"
        elif ratio >= 0.50:
            states[name] = "bent"
        else:
            states[name] = "closed"
    return states


def _extended_finger_count(arr):
    if arr is None:
        return 0
    fs = _finger_states(arr)
    return sum(1 for k, v in fs.items() if k != "thumb" and v == "straight")


# ─── TWO HAND ANALYSER ───────────────────────────────────────────────────────

class TwoHandAnalyzer:
    """
    Stateful analyser accumulating inter-frame observations to classify
    the dominant two-handed relationship token for each HamNoSys category.
    """

    def __init__(self, history_len=500):
        self._history  = deque(maxlen=history_len)
        self._seq_buf  = deque(maxlen=10)

    def analyse_frame(self, left_arr, right_arr):
        result = {
            "symmetry":   None,
            "sequencing": None,
            "role":       None,
            "hold":       None,
            "sec_shape":  None,
            "sec_loc":    None,
        }

        both_present = (left_arr is not None) and (right_arr is not None)
        only_right   = (right_arr is not None) and (left_arr is None)

        # 1. Symmetry
        if both_present:
            lw = _wrist(left_arr)
            rw = _wrist(right_arr)
            x_mid      = (lw[0] + rw[0]) / 2.0
            x_sym_err  = abs((lw[0] - x_mid) + (rw[0] - x_mid))
            y_diff     = abs(lw[1] - rw[1])
            lef        = _extended_finger_count(left_arr)
            ref        = _extended_finger_count(right_arr)
            finger_match = abs(lef - ref) <= 1

            if x_sym_err < 0.08 and y_diff < 0.10 and finger_match:
                result["symmetry"] = "hamsymmlr"
            elif x_sym_err < 0.15 and y_diff < 0.15 and finger_match:
                result["symmetry"] = "hamsymmr"

        # 2. Sequencing
        if both_present:
            dist_h = _dist(_palm_center(left_arr), _palm_center(right_arr))
            if dist_h < 0.05:
                result["sequencing"] = "hamseqatloc"
            else:
                result["sequencing"] = "hamparallel"
        elif only_right or (left_arr is not None and right_arr is None):
            result["sequencing"] = "hamsequential"

        # 3. Passive Hand Shape Preset
        passive_arr = left_arr if both_present else None
        if passive_arr is not None:
            n_ext = _extended_finger_count(passive_arr)
            fs    = _finger_states(passive_arr)
            all_closed = all(v == "closed" for k, v in fs.items() if k != "thumb")
            if all_closed:
                result["sec_shape"] = "hamsecshape0"
            elif n_ext == 1:
                result["sec_shape"] = "hamsecshape2"
            elif n_ext == 2:
                result["sec_shape"] = "hamsecshape3"
            elif n_ext == 3:
                result["sec_shape"] = "hamsecshape4"
            elif n_ext == 4:
                result["sec_shape"] = "hamsecshape5"
            elif n_ext == 5:
                result["sec_shape"] = "hamsecshape6"
            else:
                result["sec_shape"] = "hamsecshape1"

        # 4. Hold Detection
        if both_present and len(self._seq_buf) >= 5:
            recent = list(self._seq_buf)[-5:]
            lw_seq = [r[0] for r in recent if r[0] is not None]
            rw_seq = [r[1] for r in recent if r[1] is not None]
            if len(lw_seq) >= 3 and len(rw_seq) >= 3:
                l_motion = np.std(np.array(lw_seq)[:, :2], axis=0).mean()
                r_motion = np.std(np.array(rw_seq)[:, :2], axis=0).mean()
                if l_motion < 0.008 and r_motion > 0.015:
                    result["hold"] = "hamholdsecond"

        # 5. Secondary Location Qualifier
        if both_present:
            lc = _palm_center(left_arr)
            rc = _palm_center(right_arr)
            dist_h  = _dist(lc, rc)
            z_diff_h = abs(float(lc[2] - rc[2]))
            if z_diff_h > 0.08:
                result["sec_loc"] = "hamsecloc_bar"
            elif dist_h < 0.04:
                result["sec_loc"] = "hamsecloc_dot"
            elif dist_h < 0.12:
                result["sec_loc"] = "hamsecloc_comma"

        # 6. Role: simultaneous detection from history
        if len(self._history) >= 5:
            prev_5 = list(self._history)[-5:]
            had_l = any(p.get("left_present") for p in prev_5)
            had_r = any(p.get("right_present") for p in prev_5)
            if (left_arr is None or right_arr is None) and had_l and had_r:
                result["role"] = "hamsimultaneous"

        frame_summary = {
            "left_present":  left_arr is not None,
            "right_present": right_arr is not None,
            "result":        result.copy(),
        }
        self._history.append(frame_summary)

        lw_p = _wrist(left_arr)[:3]  if left_arr  is not None else None
        rw_p = _wrist(right_arr)[:3] if right_arr is not None else None
        self._seq_buf.append((lw_p, rw_p))

        return result

    def aggregate(self):
        if not self._history:
            return _empty_result()

        symm_c  = Counter()
        seq_c   = Counter()
        hold_c  = Counter()
        shape_c = Counter()
        loc_c   = Counter()

        for frame in self._history:
            r = frame["result"]
            if r["symmetry"]:   symm_c[r["symmetry"]]   += 1
            if r["sequencing"]: seq_c[r["sequencing"]]   += 1
            if r["hold"]:       hold_c[r["hold"]]         += 1
            if r["sec_shape"]:  shape_c[r["sec_shape"]]   += 1
            if r["sec_loc"]:    loc_c[r["sec_loc"]]        += 1

        n = max(len(self._history), 1)

        symmetry_token = None
        if symm_c:
            top_s, top_n = symm_c.most_common(1)[0]
            if top_n > 0.40 * n:
                symmetry_token = top_s
        if (symm_c.get("hamsymmlr", 0) > 0.2 * n and
                symm_c.get("hamsymmr", 0) > 0.2 * n):
            symmetry_token = "hamcircular"

        seq_token   = seq_c.most_common(1)[0][0]   if seq_c   else "hamparallel"
        hold_token  = hold_c.most_common(1)[0][0]  if hold_c  and hold_c.most_common(1)[0][1] > 3 else None
        shape_tok   = shape_c.most_common(1)[0][0] if shape_c else None
        loc_tok     = loc_c.most_common(1)[0][0]   if loc_c   else None

        both_frames = sum(1 for f in self._history
                          if f["left_present"] and f["right_present"])
        is_two_handed = both_frames > 0.25 * n

        return {
            "is_two_handed":    is_two_handed,
            "symmetry_token":   symmetry_token,
            "sequencing_token": seq_token,
            "hold_token":       hold_token,
            "sec_shape_token":  shape_tok,
            "sec_loc_token":    loc_tok,
            "per_frame":        [f["result"] for f in self._history],
            "final":            _build_final(
                                    symmetry_token, seq_token,
                                    hold_token, shape_tok,
                                    loc_tok, is_two_handed
                                ),
        }


def _empty_result():
    return {
        "is_two_handed":    False,
        "symmetry_token":   None,
        "sequencing_token": "hamparallel",
        "hold_token":       None,
        "sec_shape_token":  None,
        "sec_loc_token":    None,
        "per_frame":        [],
        "final":            None,
    }


def _build_final(symmetry, sequencing, hold, sec_shape, sec_loc, is_two_handed):
    if not is_two_handed:
        return None
    parts = []
    if symmetry:
        parts.append(symmetry)
    elif sequencing and sequencing != "hamparallel":
        parts.append(sequencing)
    if hold:
        parts.append(hold)
    if sec_shape:
        parts.extend(["hamsecleftpar", sec_shape, "hamsecpar"])
    if sec_loc:
        parts.append(sec_loc)
    return " ".join(parts) if parts else "hamparallel"


# ─── PUBLIC MODULE API ────────────────────────────────────────────────────────

def run_two_hands_module(video_path):
    """
    Main entry point — same pattern as all other module wrappers.

    Returns
    -------
    dict:
        is_two_handed    : bool
        symmetry_token   : str | None
        sequencing_token : str
        hold_token       : str | None
        sec_shape_token  : str | None
        sec_loc_token    : str | None
        per_frame        : list[dict]
        final            : str | None
    """
    analyzer = TwoHandAnalyzer()

    if _USE_SHARED:
        frames_data = get_video_landmarks(video_path)
        for fd in frames_data:
            left_arr  = _lm_to_arr(fd.get("left_hand"))
            right_arr = _lm_to_arr(fd.get("right_hand"))
            analyzer.analyse_frame(left_arr, right_arr)
        return analyzer.aggregate()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[TwoHands] WARNING: Cannot open video: {video_path}")
        return _empty_result()

    with _mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands_det:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands_det.process(rgb)

            left_arr  = None
            right_arr = None

            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, handed in zip(res.multi_hand_landmarks,
                                      res.multi_handedness):
                    label = handed.classification[0].label
                    arr   = np.array(
                        [(lp.x, lp.y, lp.z) for lp in lm.landmark],
                        dtype=np.float32,
                    )
                    if label == "Right" and right_arr is None:
                        right_arr = arr
                    elif label == "Left" and left_arr is None:
                        left_arr = arr

            analyzer.analyse_frame(left_arr, right_arr)

    cap.release()
    return analyzer.aggregate()


# ─── PIPELINE INTEGRATION HELPER ─────────────────────────────────────────────

def inject_two_hands(hamnosys_sequence_list, two_hands_result):
    """
    Insert two-handed tokens into the token sequence produced by
    integration_pipeline.combine_hamnosys().

    HamNoSys grammar positions:
      Symmetry operator  -> BEFORE base handshape  (slot 0)
      hamholdsecond      -> AFTER the movement      (end)
      hamsec* brackets   -> surround passive desc   (after body location)
      sequencing tokens  -> before base handshape   (slot 0)

    Parameters
    ----------
    hamnosys_sequence_list : list[str]
    two_hands_result       : dict  (output of run_two_hands_module)

    Returns
    -------
    list[str]  updated token list
    """
    if not two_hands_result.get("is_two_handed"):
        return hamnosys_sequence_list

    seq   = list(hamnosys_sequence_list)
    symm  = two_hands_result.get("symmetry_token")
    seqt  = two_hands_result.get("sequencing_token")
    hold  = two_hands_result.get("hold_token")
    shape = two_hands_result.get("sec_shape_token")
    loc   = two_hands_result.get("sec_loc_token")

    # 1. Symmetry / sequencing at slot 0
    if symm and symm not in seq:
        seq.insert(0, symm)
    elif seqt and seqt not in seq and seqt != "hamparallel":
        seq.insert(0, seqt)

    # 2. Passive hand shape after body-location token
    if shape:
        loc_kw = ("hamhead", "hamforehead", "hamchin", "hamlips",
                  "hamchest", "hamshoulders", "hamneck", "hamear",
                  "hampalm", "hamhandback")
        insert_after = None
        for i, tok in enumerate(seq):
            if any(tok.startswith(kw) for kw in loc_kw):
                insert_after = i
        if insert_after is not None:
            for j, t in enumerate(["hamsecleftpar", shape, "hamsecpar"]):
                seq.insert(insert_after + 1 + j, t)
        else:
            seq.extend(["hamsecleftpar", shape, "hamsecpar"])

    # 3. Secondary location qualifier before movement
    if loc and loc not in seq:
        mov_kw = ("hammove", "hamswing", "hamrepeat", "hamcircle",
                  "hamflex", "hamnod")
        insert_before = len(seq)
        for i, tok in enumerate(seq):
            if any(tok.startswith(kw) for kw in mov_kw):
                insert_before = i
                break
        seq.insert(insert_before, loc)

    # 4. hamholdsecond at end
    if hold and hold not in seq:
        seq.append(hold)

    return seq


# ─── STANDALONE DEMO ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Two-Handed Signs Module -- standalone demo"
    )
    parser.add_argument(
        "video", nargs="?", default="Prompt_1.mp4",
        help="Path to input sign-language video (default: Prompt_1.mp4)"
    )
    args = parser.parse_args()

    print(f"\n[TwoHands] Processing: {args.video}")
    result = run_two_hands_module(args.video)

    print("\n========== TWO-HANDED MODULE OUTPUT ==========")
    print(f"  Is two-handed       : {result['is_two_handed']}")
    print(f"  Symmetry token      : {result['symmetry_token']}")
    print(f"  Sequencing token    : {result['sequencing_token']}")
    print(f"  Hold token          : {result['hold_token']}")
    print(f"  Passive shape token : {result['sec_shape_token']}")
    print(f"  Sec location token  : {result['sec_loc_token']}")
    print(f"\n  FINAL token string  : {result['final']}")
    print("===============================================\n")
