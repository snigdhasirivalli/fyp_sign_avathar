#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install --no-cache-dir numpy==2.0.2


# In[ ]:


#!pip install --no-cache-dir mediapipe==0.10.20


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('cp "/content/drive/MyDrive/Colab Notebooks/Integration/"*.ipynb /content/')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('jupyter nbconvert --to python Handshape_Model.ipynb')
get_ipython().system('jupyter nbconvert --to python ori_model2.ipynb')
get_ipython().system('jupyter nbconvert --to python upper_body_locations_video.ipynb')
get_ipython().system('jupyter nbconvert --to python Head_and_face_location.ipynb')
get_ipython().system('jupyter nbconvert --to python hand_location_video_P.ipynb')
get_ipython().system('jupyter nbconvert --to python fing_locations_d.ipynb')
get_ipython().system('jupyter nbconvert --to python contact_types_e.ipynb')
get_ipython().system('jupyter nbconvert --to python Arm_and_Space_positions.ipynb')
get_ipython().system('jupyter nbconvert --to python movement1_prava.ipynb')
get_ipython().system('jupyter nbconvert --to python Movement_2.ipynb')


# In[ ]:


import sys
sys.path.append('/content')


# In[ ]:


import numpy as np
import cv2
import mediapipe as mp


# In[ ]:


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


# In[ ]:


mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# In[ ]:


video_path = "/content/drive/MyDrive/Colab Notebooks/Integration/Prompt_1.mp4"
output_video ="/content/output_video.mp4"


# In[ ]:


print("Running modules...")

handshape = run_handshape_module(video_path)
orientation = run_orientation_module(video_path)

upper_body = run_upper_body_location_module(video_path)
head_face = run_head_face_location_module(video_path)

hand_location = run_hand_location_module(video_path)
finger_location = run_finger_location_module(video_path)

contact = run_contact_type_module(video_path)
arm_space = run_arm_space_module(video_path)

movement1 = run_movement1_module(video_path)
movement2 = run_movement2_module(video_path)


# In[ ]:


def extract_labels(module_output):
    """
    Extract useful HamNoSys labels from module outputs
    """

    if isinstance(module_output, dict):

        # Most modules store labels here
        if "per_frame" in module_output:
            labels = module_output["per_frame"]

            # Flatten tuples and remove None
            cleaned = []
            for item in labels:

                if item is None:
                    continue

                if isinstance(item, tuple):
                    cleaned.extend([str(x) for x in item if x is not None])
                else:
                    cleaned.append(str(item))

            return cleaned

    # If already string
    if isinstance(module_output, str):
        return [module_output]

    return []


# In[ ]:


def combine_hamnosys(*modules):

    final = []

    for m in modules:
        final.extend(extract_labels(m))

    # Remove duplicates but keep order
    seen = set()
    clean = []

    for x in final:
        if x not in seen:
            clean.append(x)
            seen.add(x)

    return " ".join(clean)


# In[ ]:


def generate_hamnosys(video_path):

    '''
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
    '''
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


    hamnosys_code = combine_hamnosys(
        handshape,
        orientation,
        upper_body,
        head_face,
        hand_location,
        finger_location,
        contact,
        arm_space,
        movement1,
        movement2
    )

    print("\n========== FINAL HAMNOSYS ==========")
    print(hamnosys_code)

    return hamnosys_code


# In[ ]:


hamnosys_output = generate_hamnosys(video_path)


# In[ ]:


#output_video ="/content/drive/MyDrive/Colab Notebooks/Integration/output_pi.mp4"


# In[ ]:


# =============================
# CREATE VIDEO WITH ALL MODULE OUTPUTS
# =============================

import cv2

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

ret, frame = cap.read()
h, w, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 25, (w, h))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# =============================
# SAFE LABEL
# =============================

def safe(x):
    if x is None:
        return "none"
    return str(x)


# =============================
# FRAME EXTRACTOR
# =============================

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


# =============================
# EXTRACT FRAME DATA
# =============================

handshape_frames = get_frames(handshape, total_frames)
orientation_frames = get_frames(orientation, total_frames)

upper_frames = get_frames(upper_body, total_frames)
head_frames = get_frames(head_face, total_frames)
hand_frames = get_frames(hand_location, total_frames)
finger_frames = get_frames(finger_location, total_frames)
contact_frames = get_frames(contact, total_frames)
arm_frames = get_frames(arm_space, total_frames)

movement1_frames = get_frames(movement1, total_frames)
movement2_frames = get_frames(movement2, total_frames)


# =============================
# VIDEO PROCESSING
# =============================

frame_id = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # ---------- FRAME LABELS ----------

    handshape_label = safe(handshape_frames[frame_id])
    orientation_label = safe(orientation_frames[frame_id])

    # combine ALL location modules
    location_label = " ".join([
        safe(upper_frames[frame_id]),
        safe(head_frames[frame_id]),
        safe(hand_frames[frame_id]),
        safe(finger_frames[frame_id]),
        safe(contact_frames[frame_id]),
        safe(arm_frames[frame_id])
    ])

    movement_label = " ".join([
        safe(movement1_frames[frame_id]),
        safe(movement2_frames[frame_id])
    ])


    # =============================
    # DRAW TEXT
    # =============================

    cv2.putText(frame,
                f"Handshape: {handshape_label}",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)

    cv2.putText(frame,
                f"Orientation: {orientation_label}",
                (30,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)

    cv2.putText(frame,
                f"Location: {location_label}",
                (30,110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)

    cv2.putText(frame,
                f"Movement: {movement_label}",
                (30,150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)

    out.write(frame)

    frame_id += 1


cap.release()
out.release()

print("Video saved:", output_video)


# In[ ]:


def process_video(video_path):

    hamnosys = generate_hamnosys(video_path)

    results = {
        "hamnosys": hamnosys,
        "output_video": "/content/output_video.mp4"
    }

    return results

