"""Shared helpers for the meme matcher."""
import json
import os
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
CONFIG_PATH = os.path.join(ROOT, "memes_config.json")
LIBRARY_PATH = os.path.join(ROOT, "memes.json")
PROFILE_PATH = os.path.join(ROOT, "profile.json")  # your calibration + trained faces

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

# All 52 blendshapes MediaPipe's Face Landmarker outputs (plus "_neutral").
ALL_BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight", "tongueOut",
]

# Features that are noisy or irrelevant for "which meme face is this?":
#   - gaze direction (eyeLook*): where your eyes point isn't the expression
#   - left/right shifts of the jaw/mouth: mostly asymmetry noise
#   - mouthClose: fires whenever the jaw is open but lips are relaxed
EXCLUDED = {
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "jawLeft", "jawRight", "mouthLeft", "mouthRight", "mouthClose",
}
KEYS = [k for k in ALL_BLENDSHAPES if k not in EXCLUDED]

# Per-feature weights. Anything not listed is 1.0.
# Tweak these if one meme keeps "winning" too often.
WEIGHTS = {
    "jawOpen": 1.5,
    "browInnerUp": 1.2,
    "eyeWideLeft": 1.3,
    "eyeWideRight": 1.3,
    "eyeBlinkLeft": 0.6,   # natural blinks shouldn't trigger the wince meme
    "eyeBlinkRight": 0.6,
    "mouthSmileLeft": 1.2,
    "mouthSmileRight": 1.2,
}
WEIGHT_VEC = np.array([WEIGHTS.get(k, 1.0) for k in KEYS], dtype=np.float64)


def to_array(vec_dict):
    """dict of blendshape name -> score  ->  weighted numpy vector (len KEYS)."""
    raw = np.array([float(vec_dict.get(k, 0.0)) for k in KEYS], dtype=np.float64)
    return raw


def weighted(raw):
    return raw * WEIGHT_VEC


def download_model(url, path):
    """Download a MediaPipe model file if it isn't already on disk."""
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {os.path.basename(path)} (one-time)...")

    def hook(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            sys.stdout.write(f"\r  {pct}%")
            sys.stdout.flush()

    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp, hook)
    except Exception as e:  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        raise SystemExit(
            f"\nCould not download the model: {e}\n"
            f"Download it manually from:\n  {url}\n"
            f"and save it as:\n  {path}"
        )
    os.replace(tmp, path)
    print("\n  done.")
    return path


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["memes"]


def load_library():
    with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["memes"]


def load_profile():
    """Your saved neutral face + personally trained meme faces ({} if none)."""
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_profile(profile):
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
