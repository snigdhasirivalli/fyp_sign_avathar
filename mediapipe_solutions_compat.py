"""
mediapipe_solutions_compat.py
-----------------------------
Compatibility shim for mediapipe >= 0.10.30 on Python 3.13+.
Adds mp.solutions.hands and mp.solutions.pose using the new Tasks API,
so all existing code using mp.solutions works without any modification.

Usage: import this file BEFORE importing any module that uses mp.solutions
"""

import os
import sys
import types
import urllib.request

# Lightweight mock for tensorflow doc_controls to prevent Python 3.13 MemoryError
if 'tensorflow' not in sys.modules:
    _tf_mod = types.ModuleType('tensorflow')
    _tf_tools = types.ModuleType('tensorflow.tools')
    _tf_docs = types.ModuleType('tensorflow.tools.docs')
    class _DocControls:
        @staticmethod
        def do_not_generate_docs(func): return func
        @staticmethod
        def doc_in_current_and_submodules(func): return func
        @staticmethod
        def header_in_current_and_submodules(func): return func
    _tf_docs.doc_controls = _DocControls
    _tf_tools.docs = _tf_docs
    _tf_mod.tools = _tf_tools
    sys.modules['tensorflow'] = _tf_mod
    sys.modules['tensorflow.tools'] = _tf_tools
    sys.modules['tensorflow.tools.docs'] = _tf_docs

import mediapipe as mp

# ─────────────────────────────────────────────
# MODEL FILE DOWNLOAD
# ─────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mp_models")
os.makedirs(MODELS_DIR, exist_ok=True)

HAND_MODEL_PATH      = os.path.join(MODELS_DIR, "hand_landmarker.task")
POSE_MODEL_PATH      = os.path.join(MODELS_DIR, "pose_landmarker.task")
FACE_DET_MODEL_PATH  = os.path.join(MODELS_DIR, "face_detector.tflite")
FACE_LAND_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

HAND_MODEL_URL      = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL      = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
FACE_DET_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
FACE_LAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def _download_if_needed(path, url, name):
    if not os.path.exists(path):
        print(f"[mediapipe_compat] Downloading {name} model (~12 MB)...")
        urllib.request.urlretrieve(url, path)
        print(f"[mediapipe_compat] Saved to: {path}")


# ─────────────────────────────────────────────
# LANDMARK WRAPPER CLASSES
# (mimic old NormalizedLandmarkList / NormalizedLandmark)
# ─────────────────────────────────────────────

class _NormalizedLandmark:
    __slots__ = ("x", "y", "z", "visibility")
    def __init__(self, lm):
        self.x = lm.x
        self.y = lm.y
        self.z = lm.z
        self.visibility = getattr(lm, "visibility", 0.0)


class _LandmarkList:
    """Mimics old NormalizedLandmarkList — supports .landmark[i]"""
    def __init__(self, landmarks):
        self.landmark = [_NormalizedLandmark(lm) for lm in landmarks]

    def __iter__(self):
        return iter(self.landmark)


# ─────────────────────────────────────────────
# RESULT WRAPPERS
# ─────────────────────────────────────────────

class _Classification:
    def __init__(self, cat):
        # Tasks API uses category_name for "Left"/"Right" (or display_name, but category_name is safer)
        self.label = cat.category_name
        self.score = cat.score

class _Handedness:
    def __init__(self, categories):
        self.classification = [_Classification(c) for c in categories]

class _HandsResult:
    def __init__(self, task_result):
        if task_result.hand_landmarks:
            self.multi_hand_landmarks = [
                _LandmarkList(h) for h in task_result.hand_landmarks
            ]
            self.multi_handedness = [
                _Handedness(h) for h in task_result.handedness
            ]
        else:
            self.multi_hand_landmarks = None
            self.multi_handedness = None


class _PoseResult:
    def __init__(self, task_result):
        if task_result.pose_landmarks:
            self.pose_landmarks = _LandmarkList(task_result.pose_landmarks[0])
        else:
            self.pose_landmarks = None


# Face detection result wrappers
class _RelativeBoundingBox:
    def __init__(self, xmin, ymin, width, height):
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.height = height

class _LocationData:
    def __init__(self, bbox):
        self.relative_bounding_box = bbox

class _FaceDetectionObj:
    """Wraps a new-API Detection to look like old detection with location_data"""
    def __init__(self, det, img_w, img_h):
        bb = det.bounding_box
        self.location_data = _LocationData(_RelativeBoundingBox(
            xmin=bb.origin_x / img_w,
            ymin=bb.origin_y / img_h,
            width=bb.width   / img_w,
            height=bb.height / img_h
        ))

class _FaceDetectionResult:
    def __init__(self, task_result, img_w, img_h):
        if task_result.detections:
            self.detections = [_FaceDetectionObj(d, img_w, img_h) for d in task_result.detections]
        else:
            self.detections = []


class _FaceMeshResult:
    def __init__(self, task_result):
        if task_result.face_landmarks:
            self.multi_face_landmarks = [_LandmarkList(f) for f in task_result.face_landmarks]
        else:
            self.multi_face_landmarks = None


# ─────────────────────────────────────────────
# HANDS WRAPPER
# ─────────────────────────────────────────────

class Hands:
    """
    Drop-in replacement for mp.solutions.hands.Hands using Tasks API.
    Supports both static_image_mode=True and False (VIDEO mode).
    """

    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        **kwargs
    ):
        from mediapipe.tasks import python as _mp_tasks
        from mediapipe.tasks.python import vision as _vision

        _download_if_needed(HAND_MODEL_PATH, HAND_MODEL_URL, "Hand Landmarker")

        self._static = static_image_mode
        self._frame = 0
        self._fps = 30.0

        RunningMode = _vision.RunningMode
        mode = RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO

        options = _vision.HandLandmarkerOptions(
            base_options=_mp_tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._det = _vision.HandLandmarker.create_from_options(options)

    def process(self, rgb_image):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        if self._static:
            result = self._det.detect(mp_img)
        else:
            ts = int(self._frame * 1000 / self._fps)
            result = self._det.detect_for_video(mp_img, ts)
            self._frame += 1
        return _HandsResult(result)

    def close(self):
        try:
            self._det.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────
# POSE WRAPPER
# ─────────────────────────────────────────────

class Pose:
    """
    Drop-in replacement for mp.solutions.pose.Pose using Tasks API.
    """

    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        **kwargs
    ):
        from mediapipe.tasks import python as _mp_tasks
        from mediapipe.tasks.python import vision as _vision

        _download_if_needed(POSE_MODEL_PATH, POSE_MODEL_URL, "Pose Landmarker")

        self._static = static_image_mode
        self._frame = 0
        self._fps = 30.0

        RunningMode = _vision.RunningMode
        mode = RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO

        options = _vision.PoseLandmarkerOptions(
            base_options=_mp_tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=mode,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._det = _vision.PoseLandmarker.create_from_options(options)

    def process(self, rgb_image):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        if self._static:
            result = self._det.detect(mp_img)
        else:
            ts = int(self._frame * 1000 / self._fps)
            result = self._det.detect_for_video(mp_img, ts)
            self._frame += 1
        return _PoseResult(result)

    def close(self):
        try:
            self._det.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────
# FACE DETECTION WRAPPER
# ─────────────────────────────────────────────

class FaceDetection:
    """Drop-in for mp.solutions.face_detection.FaceDetection"""

    def __init__(self, model_selection=0, min_detection_confidence=0.5, **kwargs):
        from mediapipe.tasks import python as _mp_tasks
        from mediapipe.tasks.python import vision as _vision

        _download_if_needed(FACE_DET_MODEL_PATH, FACE_DET_MODEL_URL, "Face Detector")

        options = _vision.FaceDetectorOptions(
            base_options=_mp_tasks.BaseOptions(model_asset_path=FACE_DET_MODEL_PATH),
            running_mode=_vision.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        self._det = _vision.FaceDetector.create_from_options(options)

    def process(self, rgb_image):
        h, w = rgb_image.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self._det.detect(mp_img)
        return _FaceDetectionResult(result, w, h)

    def close(self):
        try: self._det.close()
        except Exception: pass

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


# ─────────────────────────────────────────────
# FACE MESH WRAPPER
# ─────────────────────────────────────────────

class FaceMesh:
    """Drop-in for mp.solutions.face_mesh.FaceMesh"""

    def __init__(self, static_image_mode=False, max_num_faces=1,
                 refine_landmarks=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, **kwargs):
        from mediapipe.tasks import python as _mp_tasks
        from mediapipe.tasks.python import vision as _vision

        _download_if_needed(FACE_LAND_MODEL_PATH, FACE_LAND_MODEL_URL, "Face Landmarker")

        self._static = static_image_mode
        self._frame = 0
        self._fps = 30.0

        RunningMode = _vision.RunningMode
        mode = RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO

        options = _vision.FaceLandmarkerOptions(
            base_options=_mp_tasks.BaseOptions(model_asset_path=FACE_LAND_MODEL_PATH),
            running_mode=mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._det = _vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb_image):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        if self._static:
            result = self._det.detect(mp_img)
        else:
            ts = int(self._frame * 1000 / self._fps)
            result = self._det.detect_for_video(mp_img, ts)
            self._frame += 1
        return _FaceMeshResult(result)

    def close(self):
        try: self._det.close()
        except Exception: pass

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


# ─────────────────────────────────────────────
# DRAWING UTILS (stub - not needed for pipeline)
# ─────────────────────────────────────────────

class _DrawingUtils:
    def draw_landmarks(self, *args, **kwargs): pass
    def draw_detection(self, *args, **kwargs): pass


# ─────────────────────────────────────────────
# PATCH mp.solutions
# ─────────────────────────────────────────────

class _HandsModule:
    Hands = Hands
    HAND_CONNECTIONS = []
    HAND_PALM_CONNECTIONS = []

class _PoseLandmark:
    """Mimics mp.solutions.pose.PoseLandmark — integer indices into landmark[]"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

class _PoseModule:
    Pose = Pose
    POSE_CONNECTIONS = []
    PoseLandmark = _PoseLandmark

class _FaceDetectionModule:
    FaceDetection = FaceDetection

class _FaceMeshModule:
    FaceMesh = FaceMesh
    FACEMESH_TESSELATION = []
    FACEMESH_CONTOURS = []
    FACEMESH_IRISES = []

class _SolutionsNamespace:
    hands          = _HandsModule
    pose           = _PoseModule
    face_detection = _FaceDetectionModule
    face_mesh      = _FaceMeshModule
    drawing_utils  = _DrawingUtils()
    drawing_styles = type("drawing_styles", (), {
        "get_default_hand_landmarks_style":  lambda: None,
        "get_default_pose_landmarks_style":  lambda: None,
        "get_default_face_mesh_contours_style": lambda: None,
    })()


# Inject into mediapipe module
mp.solutions = _SolutionsNamespace()

print("[mediapipe_compat] mp.solutions patched OK (mediapipe", mp.__version__, "-> Tasks API)")
