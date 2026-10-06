"""Builds memes.json: reads each meme image, extracts its facial-expression
vector with MediaPipe, and combines it with the hand-tagged vectors in
memes_config.json.

Run it directly to rebuild:   python build_library.py
(app.py also runs it automatically if memes.json doesn't exist.)
"""
import json
import os

import cv2
import numpy as np

from common import (
    FACE_MODEL_PATH, FACE_MODEL_URL, KEYS, LIBRARY_PATH, ROOT,
    download_model, load_config,
)


def make_detector():
    import mediapipe as mp  # noqa: F401  (imported lazily so errors are clear)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    download_model(FACE_MODEL_URL, FACE_MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=1,
        # Meme screenshots are small/blurry, so be lenient about detection.
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_blendshapes(detector, img_bgr):
    """Try the image as-is, upscaled, padded, and upscaled+padded."""
    import mediapipe as mp

    h, w = img_bgr.shape[:2]
    variants = [img_bgr]
    if min(h, w) < 512:
        s = 512 / min(h, w)
        up = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        variants.append(up)
    else:
        up = img_bgr
    for base in (img_bgr, up):
        pad = int(0.25 * max(base.shape[:2]))
        variants.append(cv2.copyMakeBorder(base, pad, pad, pad, pad, cv2.BORDER_REPLICATE))

    for v in variants:
        rgb = np.ascontiguousarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if result.face_blendshapes:
            return {b.category_name: float(b.score) for b in result.face_blendshapes[0]}
    return None


def build():
    config = load_config()
    needs_detector = any(m["mode"] in ("auto", "blend") for m in config)
    detector = make_detector() if needs_detector else None

    out = []
    print("\nBuilding meme library...")
    for m in config:
        entry = {"id": m["id"], "label": m["label"], "image": m["image"]}
        manual = {k: float(m.get("manual_vector", {}).get(k, 0.0)) for k in KEYS}

        if m["mode"] == "gesture":
            entry.update(type="gesture", gesture=m.get("gesture", "hands_up"), source="gesture")
            print(f"  {m['id']:<10} gesture-triggered")
            out.append(entry)
            continue

        detected = None
        if m["mode"] in ("auto", "blend"):
            img = cv2.imread(os.path.join(ROOT, m["image"]))
            if img is None:
                raise SystemExit(f"Can't read image: {m['image']}")
            detected = detect_blendshapes(detector, img)
            if detected is not None:
                detected = {k: detected.get(k, 0.0) for k in KEYS}

        if m["mode"] == "manual" or (detected is None and m["mode"] in ("auto", "blend")):
            vec, source = manual, "manual"
            if m["mode"] != "manual":
                print(f"  {m['id']:<10} NO FACE FOUND -> using hand-tagged vector")
            else:
                print(f"  {m['id']:<10} using hand-tagged vector")
        elif m["mode"] == "blend":
            vec = {k: 0.5 * detected[k] + 0.5 * manual[k] for k in KEYS}
            source = "blend"
            print(f"  {m['id']:<10} detected + hand-tagged blend")
        else:
            vec, source = detected, "detected"
            print(f"  {m['id']:<10} detected from image")

        top = sorted(vec.items(), key=lambda kv: -kv[1])[:4]
        print("             top:", ", ".join(f"{k}={v:.2f}" for k, v in top))
        if m.get("gesture"):        # face meme that a hand pose can also trigger
            entry["gesture"] = m["gesture"]
        entry.update(type="face", source=source, vector={k: round(v, 4) for k, v in vec.items()})
        out.append(entry)

    with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
        json.dump({"memes": out}, f, indent=2)
    print(f"\nSaved {LIBRARY_PATH}\n")


if __name__ == "__main__":
    build()
