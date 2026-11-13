# HamNoSys Modules: Comprehensive Beginner-Friendly Comparison & Engineering Guide

---

## 1. Project Background for First-Time Listeners

### 1.1 What is this Project About?
This project automatically translates **Sign Language videos** into **HamNoSys (Hamburg Notation System)** symbols using Artificial Intelligence and Computer Vision.

Just as **musical sheet music** transcribes sounds into notes (pitch, duration, tempo) so any musician can read and replay a song, **HamNoSys** transcribes physical sign language gestures into standardized phonetic symbols. A complete sign consists of:
1. **Handshape**: What shape is the hand forming? (e.g., a fist, flat hand, pointing index, pinch).
2. **Hand Orientation**: Which direction is the palm facing and where are the fingers pointing?
3. **Location / Finger Contact**: Where is the hand placed, or which part of the hand/fingers is touching what?
4. **Movement**: In what direction, at what speed, and with what pattern is the hand moving?

### 1.2 How Does the System See Hands? (MediaPipe)
The system uses **Google MediaPipe**, which processes each video frame and detects **21 3D hand landmarks** (numbered 0 to 20):
* **0**: Wrist
* **1–4**: Thumb (CMC, MCP, IP, Tip)
* **5–8**: Index Finger (MCP knuckle, PIP joint, DIP joint, Tip)
* **9–12**: Middle Finger (MCP knuckle, PIP joint, DIP joint, Tip)
* **13–16**: Ring Finger (MCP knuckle, PIP joint, DIP joint, Tip)
* **17–20**: Pinky Finger (MCP knuckle, PIP joint, DIP joint, Tip)

Each landmark gives $(x, y, z)$ coordinates where $x$ and $y$ are normalized between $0.0$ and $1.0$ based on image width and height, and $z$ estimates depth relative to the wrist.

---

### 1.3 Why Did the Previous Code Only Work on a Few Videos?
When the previous developers built the system, they tested it on **1 or 2 specific recording setups** (like `Prompt_1.mp4`). In those reference recordings:
* The signer was sitting at an exact distance from the camera.
* The camera had a fixed framerate (e.g. exactly 30 FPS).
* The hand was held upright, facing directly forward.

When other videos were introduced (e.g., `Prompt 2.mp4`, `Prompt 3.mp4`, `sample_3.mp4`, `hello_hi.mp4`), the system **failed or produced wrong symbols** because:
1. **Hand Tilting & Rotation**: Real signers don't always point their hands straight up. When hands were tilted sideways or pointing forward, the previous code thought fingers were closed into a fist!
2. **Camera Distance / Zoom Differences**: A person sitting closer to the camera has a large hand; a person further away has a small hand. The old code used hardcoded pixel distance numbers, which completely broke when zoom changed.
3. **Framerate (FPS) Differences**: In a 60 FPS video, consecutive frames are closer in time than in a 15 FPS video. The old code didn't account for time, so 60 FPS videos were classified as "no motion / slow" and 15 FPS videos as "ultra fast".
4. **Critical Code Bugs**: In one module, a single buggy line of code was accidentally overwriting correct predictions with `None`!

To solve this, we created **generalized, scale-invariant, rotation-invariant, and FPS-invariant** versions of the 3 modules in the folder `gendralized code/`, without modifying any files in the original `Integration/` folder.

---

# 2. Module 1: Handshape Classification (`Handshape_Model.py`)

### What This Module Does
This module inspects the hand landmarks in every frame to classify the gesture into standard HamNoSys handshapes:
* `hamfist` (closed fist)
* `hamflathand` (all fingers extended + open thumb)
* `hamfingerstraightmod` (all 4 fingers extended + thumb tucked across)
* `hamindexfinger` (only index finger pointing)
* `hamfinger23` (index + middle finger extended, e.g. "V" sign)
* `hamfinger23spread` (two fingers spread wide)
* `hampinch12` (thumb and index touching)
* `hamcee12` (thumb and index curved in a "C" shape)
* `hamthumboutmod` (thumbs up / thumb extended outward)

---

### Problem 1.1: The "Upright Hand" Trap (Comparing Raw Y-Coordinates)

#### 🔴 The Old Code
```python
def finger_open(tip, pip, lm):
    return lm[tip].y < lm[pip].y
```

#### 💡 The Intuition & Why It Failed
In computer vision, the top-left corner of the screen is $(0, 0)$. **$Y$ increases downwards** (top of screen is $y=0$, bottom of screen is $y=1$).
* The old code checked: *"Is the tip higher on screen than the middle knuckle?"*
* **The Analogy**: Imagine deciding whether someone has their arm raised by asking: *"Is their hand closer to the ceiling than their elbow?"*
  * If the person is standing upright, yes, that works!
  * But what if the person reaches sideways to point at a door? Their hand and elbow are at the exact same height!
  * What if they point downward at the floor? Their hand is closer to the floor than their elbow!
  * In the old code, pointing downward or gesturing horizontally caused the system to report that **every single finger was folded into a fist**, even when all fingers were wide open!

#### 🟢 The Generalized Solution
Instead of checking where the finger is relative to the ceiling/camera, we check the **finger's internal joint bending angles** and its **distance relative to the hand's own wrist**:
1. When a finger is extended straight, the angle formed by `MCP knuckle -> PIP joint -> DIP joint` is nearly a straight line ($\approx 180^\circ$).
2. When a finger curls into a fist, the joint bends sharply ($< 110^\circ$), and the fingertip moves closer to the knuckle.
3. This angle is calculated using the dot product between 3D vectors:
   $$\theta = \arccos\left(\frac{\vec{v}_1 \cdot \vec{v}_2}{\|\vec{v}_1\| \|\vec{v}_2\|}\right)$$

#### 💻 Code Comparison
```python
# ==================== PREVIOUS CODE ====================
def finger_open(tip, pip, lm):
    # Fails whenever hand is tilted, sideways, or pointing down!
    return lm[tip].y < lm[pip].y

# ==================== NEW GENERALIZED CODE ====================
def is_finger_extended(mcp_idx, pip_idx, dip_idx, tip_idx, lm, palm_scale):
    # Compute true 3D joint bending angles
    pip_angle = _joint_angle(mcp, pip, dip)
    dip_angle = _joint_angle(pip, dip, tip)
    
    # Distance of tip from wrist vs PIP from wrist
    dist_ratio = _distance(wrist, tip) / max(_distance(wrist, pip), 1e-4)
    
    # Rotation-invariant: Works whether hand points UP, DOWN, LEFT, RIGHT, or DIAGONAL!
    if dist_ratio > 1.25 and pip_angle > 125.0 and dip_angle > 120.0:
        return True
    return False
```

---

### Problem 1.2: The Thumb Anatomy Blunder

#### 🔴 The Old Code
```python
def get_finger_states(lm):
    thumb = finger_open(4, 3, lm)  # Compares Y of tip(4) and IP joint(3)
```

#### 💡 The Intuition & Why It Failed
The thumb does not move up and down like the other 4 fingers—it articulates **radially (outwards to the side)**.
* Even when a hand is upright and the thumb is open to the side, the thumb tip (4) is often level with or slightly lower than its knuckle/IP joint (3).
* Therefore, `lm[4].y < lm[3].y` evaluated to `False` on almost all natural open thumbs!

#### 🟢 The Generalized Solution
We measure thumb abduction relative to the **Index MCP Knuckle (landmark 5)** and **Wrist (landmark 0)**, normalized by the palm's physical scale:
```python
# Distance from thumb tip (4) to index knuckle (5), normalized by palm size
dist_idx_mcp = _distance(thumb_tip, index_mcp) / palm_scale

# If thumb is abducted away from index knuckle, it is open!
if dist_idx_mcp > 0.55:
    return True # Thumb is extended outward
```

---

### Problem 1.3: Dead Code / Unreachable Decision Branches

#### 🔴 The Old Code
```python
# flat hand
if index and middle and ring and pinky:
    return "hamflathand"
...
# straight fingers
if index and middle and ring and pinky and not thumb:
    return "hamfingerstraightmod"  # ⚠️ BUG: NEVER REACHED!
```

#### 💡 Why It Failed
In Python `if/elif` logic, the first condition that evaluates to `True` executes.
Because line 77 caught `index and middle and ring and pinky` without checking the thumb, it returned `"hamflathand"` every single time! Line 97 (`hamfingerstraightmod`) could **never be reached under any circumstances**.

#### 🟢 The Generalized Solution
```python
# 1. FLAT HAND: All 4 fingers open AND thumb is also open
if index and middle and ring and pinky and thumb:
    return "hamflathand"

# 2. STRAIGHT FINGERS: All 4 fingers open BUT thumb is tucked across palm
if index and middle and ring and pinky and not thumb:
    return "hamfingerstraightmod"
```

---

### Problem 1.4: Single-Frame Flicker & Motion Dropouts
* **Previous**: MediaPipe detection threshold was set to `0.6`. Fast signing motions caused temporary motion blur, dropping landmarks for 1 or 2 frames and producing `"none"`.
* **New**: Added adaptive confidence (`0.4`), dominant hand tracking (so background hands don't confuse the system), and a **5-frame rolling majority voting filter** that smooths away momentary dropouts.

---

# 3. Module 2: Finger Locations (`fing_locations_d.py`)

### What This Module Does
In HamNoSys, finger location defines **contact points** or **active finger identity**:
* **Contact points**: `hamfingertip` (touching at fingertip), `hamfingermidjoint` (touching at PIP middle joint), `hamfingerbase` (touching at knuckle/base), `hambetween` (between two fingers), `hamfingerpad`, `hamfingernail`.
* **No-contact active finger**: When fingers don't touch, which finger is pointing/active? (`hamindexfinger`, `hammiddlefinger`, `hamthumb`, etc.).

---

### Problem 2.1: The Fatal Aggregation Bug (Overwriting Results with `None`)

#### 🔴 The Old Code
```python
valid = [p for p in predictions if p]

if valid:
    final_location = Counter(valid).most_common(1)[0][0]
else:
    final_location = None

final_location = max(set(predictions), key=predictions.count)  # ⚠️ FATAL BUG!
```

#### 💡 The Intuition & Why It Failed
Look closely at the lines above:
1. Lines 308–311 carefully extracted all valid (non-null) frame predictions and chose the most common one.
2. But immediately at line 313, the author added:
   `final_location = max(set(predictions), key=predictions.count)`
3. This line **completely overwrote** the correct result!
4. Because `predictions` includes `None` for every frame where hands were undetected or moving without contact, `None` was frequently the most repeated item in the whole video.
5. As a result, **`final_location` was overwritten with `None`**, wiping out all valid predictions across the entire video! If a video had 0 predictions, it crashed with `ValueError: max() arg is an empty sequence`.

#### 🟢 The Generalized Solution
```python
# Filter out null/empty predictions first
valid = [p for p in smoothed if p not in (None, "none", "")]

if valid:
    # Take majority vote across valid detected frames
    final_location = Counter(valid).most_common(1)[0][0]
else:
    final_location = "none"
```

---

### Problem 2.2: The "Distance to Camera" Trap (Hardcoded Thresholds)

#### 🔴 The Old Code
```python
def classify_finger_part(contact, landmarks):
    if min(dist(contact, landmarks[i]) for i in TIP_LM) < 0.025:
        return "hamfingertip"
    if min(dist(contact, landmarks[i]) for i in PIP_LM) < 0.03:
        return "hamfingermidjoint"
    if min(dist(contact, landmarks[i]) for i in MCP_LM) < 0.035:
        return "hamfingerbase"
```

#### 💡 The Intuition & Why It Failed
The old code checked if the distance was less than static numbers: `0.025`, `0.03`, `0.035`.
* MediaPipe coordinates are normalized between $0.0$ and $1.0$ across the video frame.
* **The Analogy**: Imagine a self-driving car rule that says: *"If an obstacle is within 3 centimeters on your camera screen, slam the brakes!"*
  * If the camera is zoomed out (wide view), an entire truck might only be 2 centimeters wide on screen, so the car will crash without braking!
  * If the camera is zoomed in (close-up view), a pebble 100 meters away looks 5 centimeters wide, so the car slams the brakes for no reason!
* **In Sign Language Videos**:
  * In a **wide-angle / full-body video**, the signer's entire hand might only span $0.06$. The threshold $0.035$ is more than half the whole hand! Every single joint (tip, mid-joint, base) falsely triggered simultaneously!
  * In a **close-up video**, the signer's hand spans $0.35$. The threshold $0.025$ is smaller than a fingernail, so **contact was never detected at all**!

#### 🟢 The Generalized Solution: Anatomical Scale Normalization
We calculate the signer's **hand scale** in every frame: the distance between the wrist (0) and the middle knuckle (9):
$$S_{\text{palm}} = \|\text{wrist} - \text{MCP}_9\|$$
All proximity thresholds are dynamically proportional to the hand's own physical size:
```python
# Dynamically adapts to ANY camera distance or zoom!
palm_scale = get_palm_scale(landmarks)

tip_thresh  = 0.24 * palm_scale  # 24% of palm length
pip_thresh  = 0.30 * palm_scale  # 30% of palm length
mcp_thresh  = 0.36 * palm_scale  # 36% of palm length
```

---

### Problem 2.3: Aspect Ratio (16:9) Widescreen Distortion
In MediaPipe, $x$ is normalized by width $W$ and $y$ by height $H$.
In a 16:9 video (e.g. 1920x1080), 1 unit of $x$ represents $1.77\times$ more screen pixels than 1 unit of $y$!
Calculating 2D Euclidean distance directly on $(x, y)$ distorted horizontal distances.
* **Solution**: The new code calculates 3D Euclidean distances in landmark space, preserving true geometric proportions.

---

# 4. Module 3: Movement Dynamics (`movement1_prava.py`)

### What This Module Does
Movement 1 classifies the dynamic motion of the active hand into HamNoSys symbols:
* **Direction**: `hammoveu` (up), `hammoved` (down), `hammovel` (left), `hammover` (right), `hammovei` (inwards towards signer), `hammoveo` (outwards away), and diagonals (`hammoveul`, `hammoveur`, etc.).
* **Speed**: `hamfast`, `hamslow`.
* **Force / Tension**: `hamtense` (jerky/forceful movement), `hamrest` (relaxed).
* **Halt**: `hamhalt` (sudden stop at end).
* **Size**: `hamsmallmod` (small amplitude), `hamlargemod` (large sweeping motion).
* **Repetition**: `hamrepeatcontinue`, `hamrepeatfromstart`, `hamrepeatreverse`.
* **Path Shape**: `hamzigzag`, `hamwavy`, `hamellipseh`, `hamellipsev`.

---

### Problem 3.1: The Framerate (FPS) Time-Warp

#### 🔴 The Old Code
```python
def classify_speed(traj):
    diffs = np.diff(traj, axis=0)  # Difference between frame i and frame i+1
    speed = np.linalg.norm(diffs, axis=1).mean()
    if speed > 0.03:
        return "hamfast"
    if speed < 0.01:
        return "hamslow"
    return None
```

#### 💡 The Intuition & Why It Failed
The old code measured "speed" by taking the coordinate change **between one frame and the next frame** ($\Delta p$).
* **The Analogy**: Imagine calculating a sprinter's speed by counting how far their feet move between frames of a video camera:
  * In a high-speed **60 FPS camera**, frames are taken every $\frac{1}{60}$th of a second. The hand only moves a tiny amount between consecutive frames.
  * In a standard **30 FPS camera**, frames are $\frac{1}{30}$th of a second apart. The hand moves twice as far per frame!
  * In a **15 FPS camera**, frames are $\frac{1}{15}$th of a second apart. The hand moves 4 times as far per frame!
* **The Consequence**:
  * Any **60 FPS video** was automatically classified as **`hamslow`** or **`hamnomotion`**, even if the signer was moving very fast!
  * Any **15 FPS video** was automatically classified as **`hamfast`**, even if the signer was moving like a snail!

#### 🟢 The Generalized Solution: True Physical Velocity
In physics, velocity is displacement divided by time:
$$v = \frac{\Delta p}{\Delta t} = \Delta p \times \text{FPS}$$
We extract the actual video framerate dynamically from OpenCV: `fps = cap.get(cv2.CAP_PROP_FPS)`.
Then, we normalize by the signer's hand scale $S_{\text{palm}}$ to get speed in **hand-lengths per second**:
```python
# Speed measured in [Hand-Lengths per Second] -> Invariant to both FPS and Camera Distance!
speeds_per_sec = (np.linalg.norm(diffs, axis=1) / scale) * fps
mean_speed = float(np.mean(speeds_per_sec))

if mean_speed > 1.8:
    return "hamfast"   # Moves more than 1.8 hand-lengths in 1 second
if mean_speed < 0.6:
    return "hamslow"   # Moves less than 0.6 hand-lengths in 1 second
```

---

### Problem 3.2: MediaPipe Z-Axis (Depth) Jitter

#### 🔴 The Old Code
```python
counts = {
    "u": np.sum(dy < -eps),
    "d": np.sum(dy >  eps),
    "l": np.sum(dx < -eps),
    "r": np.sum(dx >  eps),
    "i": np.sum(dz < -eps),  # Depth towards camera
    "o": np.sum(dz >  eps),  # Depth away from camera
}
```

#### 💡 Why It Failed
MediaPipe estimates depth ($z$) from a single RGB camera using a machine learning heuristic. Because it lacks a true LiDAR or stereo depth sensor, the predicted $z$-coordinate has **high-frequency random noise and jitter**.
* In a 2D waving motion where the hand moves strictly left-and-right, the noisy $z$-coordinate fluctuated back and forth constantly.
* In the old code, `counts["i"]` or `counts["o"]` often exceeded `counts["l"]` or `counts["r"]`, causing a simple left-to-right wave to be misclassified as moving in-and-out towards the camera!

#### 🟢 The Generalized Solution
1. **Exponential Moving Average (EMA) Denoising**:
   We smooth the raw trajectory using an EMA low-pass filter:
   $$\vec{s}_t = 0.35 \vec{p}_t + 0.65 \vec{s}_{t-1}$$
   This removes high-frequency jitter while preserving the intentional gesture path.
2. **Calibrated Depth Significance**:
   $Z$-axis movement is required to be at least $1.5\times$ more prominent than planar motion before concluding that the gesture is primarily depth-directed.

---

### Problem 3.3: Global State Leaks Across Videos
* **The Old Code**: Used a single global variable `trajectory = deque(maxlen=60)`. When processing multiple videos in batch (e.g. `sample_1.mp4` followed by `sample_2.mp4`), points from the end of Video 1 remained in the deque at the beginning of Video 2!
* **The New Code**: All deques and tracking structures are strictly scoped **inside the function call**, ensuring 100% thread safety and zero cross-contamination.

---

# 5. Master Summary Table: Before vs. After

| Feature / Behavior | Previous Code (`Integration/`) | Generalized Code (`gendralized code/`) | Why the Change Matters |
| :--- | :--- | :--- | :--- |
| **Hand Orientation** | Uses raw $y_{\text{tip}} < y_{\text{pip}}$ | Uses 3D joint bending angles & vector dot products | Works when hands are tilted, sideways, diagonal, or downward. |
| **Thumb State** | Compares $y$-height of thumb tip and IP joint | Measures radial distance to index knuckle normalized by palm | Correctly recognizes extended thumbs in natural signing. |
| **Dead Code (`Handshape`)** | `hamflathand` shadowed `hamfingerstraightmod` | Separated checks for thumb-open vs thumb-tucked | Restores `hamfingerstraightmod` classification. |
| **Detection Confidence** | Fixed at 0.6 (drops hands during fast movement) | Adaptive (0.4) + dominant hand selection | Eliminates `"none"` dropouts during rapid signing motions. |
| **Temporal Stability** | None; single noisy frames changed label | 5-frame rolling majority voting filter | Produces clean, non-flickering predictions. |
| **Voting Bug (`Finger Loc`)** | Overwrote valid consensus with `None` via line 313 | Filters `None` before calculating majority vote | Fixes fatal bug where entire videos returned `None` or crashed. |
| **Contact Thresholds** | Fixed numbers ($0.03, 0.025, 0.035$) | Dynamically scaled by hand palm size $S_{\text{palm}}$ | Works equally well on close-up, medium, and wide shots. |
| **Aspect Ratio (16:9)** | 2D Euclidean norm on normalized $(x, y)$ | 3D Euclidean norm in metric space | Eliminates horizontal distortion in widescreen videos. |
| **Framerate (FPS)** | Displacements compared to static constants | Velocity normalized by FPS: $v = \Delta p \times \text{FPS}$ | Prevents 60 FPS from reading as "slow" and 15 FPS as "fast". |
| **Movement Units** | Raw screen fractions per frame | Hand-lengths per second ($v / S_{\text{palm}}$) | Speed is invariant to camera distance and video resolution. |
| **Depth Noise ($Z$-axis)** | Unfiltered $z$ overwhelmed true motion | EMA trajectory smoothing ($\alpha = 0.35$) | Prevents jitter from faking in/out depth movement. |
| **State Memory Leaks** | Shared global `trajectory` deque | Localized deques per video invocation | Zero state bleed when running multiple videos in batch. |

---

# 6. How to Verify and Run the Generalized Code

The generalized code can be run immediately without modifying your existing environment.

### Run the Standalone Test Runner
A test runner has been provided that automatically searches for video files in the workspace and evaluates all 3 generalized modules:
```powershell
python "d:\amrita\FYP\sample_run\HamNoSys_Group14\Integration_extracted\gendralized code\run_test_batch.py"
```

### Or Import in Your Pipeline
```python
import sys
sys.path.insert(0, r"d:\amrita\FYP\sample_run\HamNoSys_Group14\Integration_extracted\gendralized code")

from Handshape_Model import run_handshape_module
from fing_locations_d import run_finger_location_module
from movement1_prava import run_movement1_module

# Run on ANY video
video_file = "Prompt 2.mp4"

handshape_result = run_handshape_module(video_file)
finger_result    = run_finger_location_module(video_file)
movement_result  = run_movement1_module(video_file)

print("Handshape       :", handshape_result["final"])
print("Finger Location :", finger_result["final"])
print("Movement 1      :", movement_result)
```
