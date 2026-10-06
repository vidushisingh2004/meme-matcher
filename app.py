"""Meme Matcher: your webcam face -> the meme that looks most like you.

Run:  python app.py

Keys:  q / Esc  quit        c  calibrate your neutral face
       r  reset calibration d  toggle score debug panel
       t  train: copy each meme face so it matches YOU (Esc cancels)
       x  forget trained faces   s  save a snapshot   n  next camera
"""
import argparse
import datetime
import os
import time

import cv2
import numpy as np

from common import (
    FACE_MODEL_PATH, FACE_MODEL_URL, HAND_MODEL_PATH, HAND_MODEL_URL,
    LIBRARY_PATH, ROOT, download_model, load_library, load_profile, save_profile,
)
from gestures import GESTURES, GESTURE_PRIORITY
from matcher import TAKES, HoldFlag, Matcher

WINDOW = "Meme Matcher"
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL = 540  # height of the video, and width/height of the meme panel
FRAME_PATH = os.path.join(ROOT, "frame.jpg")   # decorative border (optional)
VIEW_W, VIEW_H = 960, PANEL                    # size of the left (camera) side
CAM_W, CAM_H = 880, 470                       # camera window inside the frame
MEME_SIZE = 460                               # meme window inside its frame


# ---------------------------------------------------------------- drawing ---
def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def fit(img, w, h, bg=(28, 28, 28)):
    """Resize img to fit inside w x h without stretching (letterboxed)."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    canvas = np.full((h, w, 3), bg, np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return canvas


def darken_bar(img, y0, y1, alpha=0.55):
    roi = img[y0:y1]
    img[y0:y1] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)


def score_bar(img, x, y, w, frac, color=(80, 220, 120)):
    cv2.rectangle(img, (x, y), (x + w, y + 12), (70, 70, 70), -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0, min(1, frac))), y + 12), color, -1)


def cover(img, w, h):
    """Resize img to fill w x h, center-cropping whatever overflows."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    img = cv2.resize(img, (max(w, round(iw * s)), max(h, round(ih * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    y, x = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    return img[y:y + h, x:x + w]


def load_frame(w, h, cw, ch):
    """Border art sized w x h, with a cw x ch window (white mat + shadow) cut in
    the middle. Returns (background, (x, y)) or None if there's no frame.jpg."""
    img = cv2.imread(FRAME_PATH)
    if img is None:
        return None
    if (img.shape[0] > img.shape[1]) == (w > h):   # match the art's orientation
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    bg = cover(img, w, h)
    x, y = (w - cw) // 2, (h - ch) // 2
    shadow = np.zeros(bg.shape[:2], np.float32)
    cv2.rectangle(shadow, (x + 6, y + 8), (x + cw + 6, y + ch + 8), 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 10)[..., None] * 0.45
    bg = (bg * (1 - shadow)).astype(np.uint8)
    b = 8                                       # white mat around the window
    cv2.rectangle(bg, (x - b, y - b), (x + cw + b, y + ch + b), (255, 255, 255), -1)
    return bg, (x, y)


def blank_panel(message, sub=None):
    p = np.full((PANEL, PANEL, 3), (28, 28, 28), np.uint8)
    (tw, _), _ = cv2.getTextSize(message, FONT, 0.9, 2)
    put_text(p, message, ((PANEL - tw) // 2, PANEL // 2), 0.9)
    if sub:
        (sw, _), _ = cv2.getTextSize(sub, FONT, 0.55, 1)
        put_text(p, sub, ((PANEL - sw) // 2, PANEL // 2 + 36), 0.55, (180, 180, 180), 1)
    return p


# ------------------------------------------------------------------ setup ---
def ensure_library(rebuild):
    if rebuild or not os.path.exists(LIBRARY_PATH):
        from build_library import build
        build()


def make_detectors(use_hands):
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    download_model(FACE_MODEL_URL, FACE_MODEL_PATH)
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        num_faces=1,
    ))
    hands = None
    if use_hands:
        download_model(HAND_MODEL_URL, HAND_MODEL_PATH)
        hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        ))
    return face, hands


def open_camera(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    return cap


class Clock:
    """MediaPipe VIDEO mode needs strictly increasing millisecond timestamps."""

    def __init__(self):
        self.last = 0

    def next(self):
        self.last = max(int(time.monotonic() * 1000), self.last + 1)
        return self.last


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description="Match your face to a meme.")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--no-hands", action="store_true", help="disable hand gestures")
    ap.add_argument("--rebuild", action="store_true", help="rebuild memes.json from the images")
    args = ap.parse_args()

    ensure_library(args.rebuild)
    library = load_library()
    meme_by_id = {m["id"]: m for m in library}
    meme_img = {}
    for m in library:
        img = cv2.imread(os.path.join(ROOT, m["image"]))
        if img is None:
            raise SystemExit(f"Can't read meme image: {m['image']}")
        meme_img[m["id"]] = fit(img, PANEL, PANEL)

    gesture_memes = {m["gesture"]: m for m in library if m.get("gesture") in GESTURES}
    use_hands = bool(gesture_memes) and not args.no_hands

    import mediapipe as mp
    face_det, hand_det = make_detectors(use_hands)
    matcher = Matcher(library, load_profile())
    hand_flags = {g: HoldFlag(on_after=3, off_after=8) for g in gesture_memes}
    face_clock, hand_clock = Clock(), Clock()

    cam_index = args.camera
    cap = open_camera(cam_index)
    if not cap.isOpened():
        raise SystemExit(
            f"Couldn't open camera {args.camera}. Close other apps using it, "
            "check OS camera permissions, or try --camera 1."
        )

    framed = load_frame(VIEW_W, VIEW_H, CAM_W, CAM_H)
    meme_framed = load_frame(PANEL, PANEL, MEME_SIZE, MEME_SIZE)
    vw_, vh_ = (CAM_W, CAM_H) if framed else (VIEW_W, VIEW_H)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    debug, toast, toast_until = False, "", 0.0
    fps, t_prev = 0.0, time.time()
    train = None   # {"queue": [meme ids], "start": when the current one begins}
    TRAIN_COUNTDOWN = 3.0
    print("Running. Make faces! (q to quit)")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost the camera feed.")
            break

        # Detect on the un-mirrored frame so left/right blendshapes stay correct.
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_res = face_det.detect_for_video(mp_img, face_clock.next())
        hand_lms = []
        if hand_det is not None:
            hand_lms = hand_det.detect_for_video(mp_img, hand_clock.next()).hand_landmarks
        # Debounce each gesture separately; the highest-priority active one wins.
        active = [g for g in GESTURE_PRIORITY if g in hand_flags
                  and hand_flags[g].update(GESTURES[g](hand_lms))]
        gesture = active[0] if active else None
        hands_active = gesture is not None

        face_found = bool(face_res.face_blendshapes)
        if face_found:
            bs = {b.category_name: b.score for b in face_res.face_blendshapes[0]}
            meme_id, score = matcher.update(bs)
        else:
            matcher.no_face()
            meme_id, score = None, 0.0

        if hands_active:                       # gesture beats face matching
            meme_id, score = gesture_memes[gesture]["id"], 1.0

        # ---- guided training: copy each meme face, we record your version ----
        if train:
            tid, take = train["queue"][0]
            if train.get("capturing") and not matcher.capturing:   # just finished
                train["queue"].pop(0)
                save_profile(matcher.profile())
                if take == TAKES - 1:
                    toast = f"Saved your {meme_by_id[tid]['label']}"
                    toast_until = time.time() + 1.5
                train["capturing"] = False
                train["start"] = time.time() + (TRAIN_COUNTDOWN if take == TAKES - 1 else 2.0)
                if not train["queue"]:
                    train = None
            elif not train.get("capturing") and time.time() >= train["start"] and face_found:
                if take == 0:
                    matcher.clear_trained(tid)    # retraining replaces old takes
                matcher.start_capture(tid)
                train["capturing"] = True
            meme_id = None

        # ---- compose the display -------------------------------------------
        view = cover(cv2.flip(frame, 1), vw_, vh_)

        if train:
            tid, take = train["queue"][0]
            panel = meme_img[tid].copy()
            darken_bar(panel, 0, 44)
            if train.get("capturing"):
                msg = "HOLD IT!"
            elif not face_found:
                msg = "Get your face in frame"
            else:
                msg = (f"Copy this face... {max(1, int(train['start'] - time.time()) + 1)}"
                       f"  ({take + 1}/{TAKES})")
            put_text(panel, msg, (14, 31), 0.85, (120, 255, 160))
            darken_bar(panel, PANEL - 40, PANEL)
            put_text(panel, f"training {meme_by_id[tid]['label']} (Esc cancels)",
                     (14, PANEL - 14), 0.55)
        elif meme_id:
            panel = meme_img[meme_id]
        elif not face_found and not hands_active:
            panel = blank_panel("No face detected", "Move into frame / add light")
        else:
            panel = blank_panel("Make a face!", "Exaggerate it - memes are extreme")

        # HUD on the video side
        darken_bar(view, 0, 30)
        put_text(view, "q quit  c calibrate  r reset  d debug  t train  n cam",
                 (10, 21), 0.5, (230, 230, 230), 1)
        if debug:
            y = 60
            put_text(view, f"FPS {fps:.0f}  gesture={gesture}", (10, y), 0.5, (255, 220, 120), 1)
            for m in matcher.faces:
                y += 24
                s = matcher.last_scores.get(m["id"], 0.0)
                put_text(view, f"{m['id']:<9}{'*' if m['id'] in matcher.user else ''}", (10, y), 0.5, (255, 255, 255), 1)
                score_bar(view, 110, y - 11, 160, s)
                put_text(view, f"{s:.2f}", (280, y), 0.5, (255, 255, 255), 1)

        if time.time() < toast_until:
            put_text(view, toast, (10, vh_ - 14), 0.7, (120, 255, 160))

        if framed:
            left = framed[0].copy()
            x, y = framed[1]
            left[y:y + CAM_H, x:x + CAM_W] = view
        else:
            left = view
        if meme_framed:
            right = meme_framed[0].copy()
            x, y = meme_framed[1]
            right[y:y + MEME_SIZE, x:x + MEME_SIZE] = cv2.resize(
                panel, (MEME_SIZE, MEME_SIZE), interpolation=cv2.INTER_AREA)
        else:
            right = panel
        canvas = np.hstack([left, right])
        cv2.imshow(WINDOW, canvas)

        # ---- keys ----------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == 27 and train:                # Esc cancels training first
            matcher.cancel_capture()
            train = None
            toast, toast_until = "Training cancelled", time.time() + 1.5
        elif key in (ord("q"), 27):
            break
        elif key == ord("c"):
            matcher.calibrate()
            save_profile(matcher.profile())
            toast, toast_until = "Neutral face saved", time.time() + 1.5
        elif key == ord("r"):
            matcher.reset_calibration()
            save_profile(matcher.profile())
            toast, toast_until = "Calibration reset", time.time() + 1.5
        elif key == ord("t") and not train:
            train = {"queue": [(m["id"], k) for m in matcher.faces for k in range(TAKES)],
                     "start": time.time() + TRAIN_COUNTDOWN}
        elif key == ord("x"):
            matcher.clear_trained()
            save_profile(matcher.profile())
            toast, toast_until = "Trained faces cleared", time.time() + 1.5
        elif key == ord("d"):
            debug = not debug
        elif key == ord("n"):                  # next camera (skips the iPhone)
            for nxt in (cam_index + 1, 0):
                new_cap = open_camera(nxt)
                if new_cap.isOpened():
                    cap.release()
                    cap, cam_index = new_cap, nxt
                    break
                new_cap.release()
            toast, toast_until = f"Camera {cam_index}", time.time() + 1.5
        elif key == ord("s"):
            os.makedirs(os.path.join(ROOT, "snapshots"), exist_ok=True)
            name = datetime.datetime.now().strftime("meme_%Y%m%d_%H%M%S.png")
            cv2.imwrite(os.path.join(ROOT, "snapshots", name), canvas)
            toast, toast_until = f"Saved snapshots/{name}", time.time() + 2.0
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - t_prev))
        t_prev = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
