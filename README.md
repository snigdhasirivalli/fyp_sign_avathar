# Avatar Generator FYP - Generalized HamNoSys Modules

This repository contains the generalized, rotation-invariant, and scale-invariant computer vision modules for translating sign language videos into standardized **HamNoSys (Hamburg Notation System)** phonetic representations.

---

## 🌟 Overview of Modules

### 1. Handshape Classification (`Handshape_Model.py`)
* **Rotation-Invariant 3D Joint Angles:** Replaces naive screen-plane Y-axis comparisons (`lm[tip].y < lm[pip].y`) with true 3D vector dot-product bending angles across MCP, PIP, and DIP joints ($\theta = \arccos\frac{\vec{v}_1 \cdot \vec{v}_2}{\|\vec{v}_1\| \|\vec{v}_2\|}$).
* **Dynamic Palm-Scale Normalization:** Normalizes all distances by the distance between wrist (landmark 0) and middle MCP (landmark 9), ensuring invariance to camera distance, zoom, and resolution.
* **Multi-Hand Dominance Tracking:** Dynamically identifies and tracks the dominant signing hand using palm bounding box area.
* **Temporal Smoothing:** 5-frame rolling majority voting window filters out single-frame tracking glitches and motion blur.

### 2. Finger Locations & Contact Detection (`fing_locations_d.py`)
* **Scale-Adaptive Proximity Thresholds:** Replaces hardcoded pixel distance thresholds with palm-scaled contact radii.
* **Rotational Contact Detection:** Accurately classifies finger pad, fingertip, and midjoint contacts regardless of hand orientation.

### 3. Movement Trajectory & Kinematics (`movement1_prava.py`)
* **FPS-Invariant Velocity Profiling:** Normalizes inter-frame displacement by real video frame rate, ensuring consistent classification across 15 FPS, 30 FPS, and 60 FPS videos.
* **Trajectory Curvature Analysis:** Robustly classifies directional movements (`hammoveu`, `hammoved`, `hammovel`, `hammover`, `hammoveo`, `hammovei`, `hamwavy`, etc.).

### 4. Hand Orientation (`ori_model2.py`)
* **Geometric View Detection:** `bird` / `right` / `signer` come from the hand's own geometry — finger long-axis depth dominance and the palm normal — instead of the wrist position, so framing and camera distance no longer change the view.
* **Roll Correction:** Screen directions are de-rotated by the shoulder-line tilt (computed from |Δx| so a frontal signer is not flipped 180°), so a tilted camera no longer rotates every direction label.
* **3D Finger Direction:** Directions average the truly unfolded index/middle/ring/pinky fingers in a palm-scaled 3D frame and map onto the HamNoSys `hamextfinger*` vocabulary; depth suffixes require the depth component to dominate.
* **Anatomical Palm Normal:** `hampalm*` comes from the palm-plane normal with its sign resolved from the signer's body side (MediaPipe handedness flips on mirrored input).
* **Resting-Hand Rejection:** A hand hanging below the hips is ignored, so it cannot dominate the temporal vote.
* **No Import-Time Training:** The four synthetic RandomForests and per-frame JPEG round-trip are gone; detection is geometric and deterministic.

### 5. Upper-Body Location (`upper_body_locations_video.py`)
* **Real Hand Landmarks:** Uses the palm centre from the hands model instead of estimating contact from the pose wrist; the pose wrist is added as a candidate when the hand model drops a shaped hand.
* **Body-Plane Geometry:** All hand-to-body distances are 2D, because pose and hand landmarks estimate depth in different reference frames.
* **Scale-Free Body Frame:** All radii are normalized by shoulder width, torso height, or limb length.
* **Segment Distances:** Arms are located with point-to-segment distance, so horizontal and diagonal arms are handled (the old rule only worked for hanging arms).
* **Torso Silhouette Gate:** `hamchest` / `hamstomach` / `hambelowstomach` require the hand inside the torso outline; hands beside the body are no longer mislabeled.
* **Both Hands Evaluated:** The most specific hand-body contact wins, instead of always preferring the left hand.

### 6. Head & Face Location (`Head_and_face_location.py`)
* **Face-Outline Regions:** The fingertip is placed inside the pose landmark face box (`hamheadtop`, `hamforehead`, `hameyes`, `hamnose`, `hamlips`, `hamchin`, `hamneck`, …) instead of fixed pixel offsets.
* **Halo Tolerance:** A fingertip up to 30% of the face height outside the box still gets a label, so a hand beside the head is located rather than dropped.
* **Nearest Fingertip:** The closest fingertip across both hands is used, with mouth motion history distinguishing `hamteeth` from `hamlips`.
* **Temporal Smoothing:** Per-frame labels are majority-filtered.

### 7. Hand Location (`hand_location_video_P.py`)
* **Palm-Local Frame:** `hamthumbball`, `hamthumbside`, `hampinkyside`, `hamhandback`, `hampalm` are resolved in a palm-local frame (thumb axis, finger axis, palm normal).
* **Interacting-Hand Direction:** The contacting side is chosen from the direction to the other hand's palm centre; a lone hand defaults to `hampalm`, with a thumb-ball override.
* **Anatomical Sides:** Left/right is inferred from body position, fixing mirrored third-person videos.
* **Contact Points:** The other hand's nearest fingertip (or the signer's own fingertip) defines the contact point, with mode smoothing.

### 8. Contact Types (`contact_types_e.py`)
* **Palm-Scaled Thresholds:** Contact radii are multiples of the palm scale, not raw MediaPipe units.
* **5-Fingertip Depth:** Interlock / cross / behind use all fingertips and signed depth, not just the index tips.
* **Body Contact:** One-handed signs touching the chest, shoulders, or face are detected via the torso silhouette and a signed depth test.
* **No Self-Touch:** A single hand can no longer produce a fake `hamtouch`.
* **Temporal History:** Cross and brushing use motion history; final label ignores `no-contact` frames when real contact exists.

### 9. Arm & Space Positions (`Arm_and_Space_positions.py`)
* **Normalized Arm Features:** Straightness, elbow angle, lateral offset and wrist height are normalized by shoulder width or torso height.
* **Rule Cleanup:** Unreachable branches and the old pixel thresholds are replaced by physical criteria for `hamarmextended`, `hamlrat`, `hamlrbeside`, `hamneutralspace`, `hamdoublebent`.

### 10. Circular & Oscillating Movement (`Movement_2.py`)
* **Palm-Centre Trajectories:** Motion is tracked from the palm centre (waving pivots at the wrist), expressed in a shoulder-width/roll-corrected frame, and linearly detrended so travelling signs are not mistaken for straight strokes.
* **Circle-Fit Loop Test:** `hamcircle*` / `hamclock*` require a hand-sized least-squares circle, a large swept angle and a small radial residual, which suppresses false loops on waves and straight signs.
* **Hand Identity Gate:** A tracked hand that jumps more than ~0.9 shoulder widths is treated as the other hand, so the resting hand cannot splice a fake loop into the trajectory.
* **FPS-Aware Oscillation:** `hamwavy` / `hamzigzag` / `hamswinging` counting uses hysteresis in shoulder-widths per second.

---

## 📊 Benchmark & Accuracy Validation

### 1. Indian Sign Language (ISL) Dataset (N=141 Videos)
Evaluated against verified ground truth from `groundtruth_check`:
* **Generalized 3D Code:** **21.28%** Top-1 Accuracy | Avg F1: **0.0609** | **24 Solo Wins** | 11 Diverse Classes
* **Original Normal Code:** 19.15% Top-1 Accuracy | Avg F1: 0.0537 | Mode Collapse: 87.9% `hamfist`
* **Neural MLP Model:** 19.15% Top-1 Accuracy | Avg F1: 0.0562 | Mode Collapse: 100% `hamfist`

### 2. German Sign Language (DGS) Dataset (N=50 Frontal Videos)
Evaluated against clean ground truth from `downloaded_dgs_dataset`:
* **Generalized 3D Code:** **40.0%** (20/50 correct) | **10 Decisive Solo Wins**
* **Original Normal Code:** 24.0% (12/50 correct)
* **Neural MLP Model:** 18.0% (9/50 correct)

---

## 🚀 Quickstart & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Any Single Module
```bash
python Handshape_Model.py path/to/video.mp4
python ori_model2.py path/to/video.mp4
python contact_types_e.py path/to/video.mp4
```

### 3. Run All Modules on All Videos
```bash
python run_test_batch.py
```

MediaPipe model files are downloaded automatically into `_mp_models/` on first run.

---

## 📁 Repository Structure
```
├── Handshape_Model.py                      # Generalized 3D Handshape Module
├── fing_locations_d.py                     # Generalized Finger Contact Module
├── movement1_prava.py                      # Generalized Movement Kinematics Module
├── ori_model2.py                           # Generalized Hand Orientation Module
├── upper_body_locations_video.py           # Generalized Upper-Body Location Module
├── Head_and_face_location.py               # Generalized Head & Face Location Module
├── hand_location_video_P.py                # Generalized Hand Location Module
├── contact_types_e.py                      # Generalized Contact-Type Module
├── Arm_and_Space_positions.py              # Generalized Arm & Space Module
├── Movement_2.py                           # Generalized Circular/Oscillation Module
├── run_test_batch.py                       # Batch evaluation harness (all modules)
├── mediapipe_solutions_compat.py          # Compatibility layer for modern MediaPipe
├── CHANGES_REPORT.md                       # Comprehensive engineering documentation
├── create_word_doc.py                      # Script to export formal Word documentation
├── HamNoSys_Module_Generalization_Report.docx  # Formatted Word report
├── requirements.txt                        # Dependency list
└── README.md                               # Project documentation
```
