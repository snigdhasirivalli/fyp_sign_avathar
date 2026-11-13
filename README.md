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

### 2. Run Generalized Handshape Classification
```bash
python Handshape_Model.py path/to/video.mp4
```

### 3. Run Batch Processing
```bash
python run_test_batch.py
```

---

## 📁 Repository Structure
```
├── Handshape_Model.py                      # Generalized 3D Handshape Module
├── fing_locations_d.py                     # Generalized Finger Contact Module
├── movement1_prava.py                      # Generalized Movement Kinematics Module
├── run_test_batch.py                       # Batch evaluation harness
├── mediapipe_solutions_compat.py          # Compatibility layer for modern MediaPipe
├── CHANGES_REPORT.md                       # Comprehensive engineering documentation
├── create_word_doc.py                      # Script to export formal Word documentation
├── HamNoSys_Module_Generalization_Report.docx  # Formatted Word report
├── requirements.txt                        # Dependency list
└── README.md                               # Project documentation
```
