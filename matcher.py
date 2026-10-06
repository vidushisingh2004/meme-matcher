"""Matching logic (no camera / MediaPipe code in here, so it's easy to test)."""
from collections import deque

import numpy as np

from common import KEYS, to_array, weighted

# --- Tuning knobs -----------------------------------------------------------
SMOOTHING_FRAMES = 6      # average this many frames to stop jitter
MIN_SCORE = 0.55          # below this, show "no match"
INTENSITY_FRACTION = 0.6  # you need ~60% of a meme's intensity for full score
SWITCH_FRAMES = 5         # a new meme must win this many frames in a row
SWITCH_MARGIN = 0.06      # ...and beat the meme on screen by this much
TAKES = 3                 # how many times you pose each meme when training
CAPTURE_FRAMES = 30       # frames recorded (median) when training a meme face
# -----------------------------------------------------------------------------


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _to_dict(vec):
    return {k: round(float(v), 4) for k, v in zip(KEYS, vec)}


class Matcher:
    def __init__(self, library, profile=None):
        profile = profile or {}
        self.faces = [m for m in library if m.get("type") == "face"]
        self.library_ref = {m["id"]: weighted(to_array(m["vector"])) for m in self.faces}
        # Your own takes of each meme face (baseline-subtracted raw blendshapes).
        # Older profiles stored one dict per meme; newer ones store a list.
        self.user = {}
        for k, v in profile.get("memes", {}).items():
            if k in self.library_ref:
                takes = v if isinstance(v, list) else [v]
                self.user[k] = [to_array(t) for t in takes]
        self.baseline = to_array(profile.get("baseline", {}))   # raw space
        self._refresh_refs()

        self.history = deque(maxlen=SMOOTHING_FRAMES)
        self.current = None       # id currently displayed
        self._cand = None
        self._cand_count = 0
        self.last_scores = {}
        self.last_raw = None
        self._capture_id = None
        self._capture_buf = []

    def _refresh_refs(self):
        """Prefer your own trained takes over the meme image's face."""
        self.ref = {mid: ([weighted(t) for t in self.user[mid]] if self.user.get(mid) else [v])
                    for mid, v in self.library_ref.items()}

    def profile(self):
        return {"baseline": _to_dict(self.baseline),
                "memes": {k: [_to_dict(t) for t in v] for k, v in self.user.items()}}

    # -- calibration ---------------------------------------------------------
    def calibrate(self):
        """Treat the current (smoothed) face as 'neutral'."""
        if self.last_raw is not None:
            self.baseline = self.last_raw.copy()

    def reset_calibration(self):
        self.baseline = np.zeros_like(self.baseline)

    # -- training your own meme faces ----------------------------------------
    @property
    def capturing(self):
        return self._capture_id is not None

    def start_capture(self, meme_id):
        self._capture_id, self._capture_buf = meme_id, []

    def cancel_capture(self):
        self._capture_id, self._capture_buf = None, []

    def clear_trained(self, meme_id=None):
        """Forget your takes for one meme, or for all of them."""
        if meme_id is None:
            self.user = {}
        else:
            self.user.pop(meme_id, None)
        self._refresh_refs()

    def _finish_capture(self):
        delta = np.clip(np.median(self._capture_buf, axis=0) - self.baseline, 0.0, None)
        self.user.setdefault(self._capture_id, []).append(delta)
        self.cancel_capture()
        self._refresh_refs()

    # -- scoring ---------------------------------------------------------------
    def score_all(self, live_w):
        """live_w: weighted live vector. Returns {meme_id: score 0..1}."""
        live_norm = float(np.linalg.norm(live_w))
        scores = {}
        for mid, refs in self.ref.items():
            best = 0.0
            for ref in refs:                 # best of your takes
                need = INTENSITY_FRACTION * float(np.linalg.norm(ref))
                intensity = min(1.0, live_norm / need) if need > 0 else 0.0
                best = max(best, max(0.0, _cos(live_w, ref)) * intensity)
            scores[mid] = best
        return scores

    def update(self, raw_blendshapes):
        """Feed one frame's blendshape dict. Returns (meme_id | None, score)."""
        raw = to_array(raw_blendshapes)
        if self.capturing:
            self._capture_buf.append(raw)
            if len(self._capture_buf) >= CAPTURE_FRAMES:
                self._finish_capture()
        self.history.append(raw)
        self.last_raw = np.mean(self.history, axis=0)
        live_w = weighted(np.clip(self.last_raw - self.baseline, 0.0, None))

        self.last_scores = self.score_all(live_w)
        best = max(self.last_scores, key=self.last_scores.get)
        best_score = self.last_scores[best]
        target = best if best_score >= MIN_SCORE else None
        cur_score = self.last_scores.get(self.current, 0.0)
        if (self.current and target not in (None, self.current)
                and cur_score >= MIN_SCORE and best_score < cur_score + SWITCH_MARGIN):
            target = self.current            # not clearly better: stay put

        # Hysteresis: only switch once the new target has won several frames.
        if target == self.current:
            self._cand, self._cand_count = None, 0
        elif target == self._cand:
            self._cand_count += 1
            if self._cand_count >= SWITCH_FRAMES:
                self.current, self._cand, self._cand_count = target, None, 0
        else:
            self._cand, self._cand_count = target, 1

        shown_score = self.last_scores.get(self.current, 0.0) if self.current else 0.0
        return self.current, shown_score

    def no_face(self):
        """Call when no face is visible this frame."""
        self.history.clear()
        self.last_raw = None
        self.last_scores = {}
        self.current, self._cand, self._cand_count = None, None, 0


class HoldFlag:
    """Turns a flickery per-frame boolean into a stable on/off signal."""

    def __init__(self, on_after=4, off_after=8):
        self.on_after, self.off_after = on_after, off_after
        self.state, self._streak = False, 0

    def update(self, value):
        if value == self.state:
            self._streak = 0
        else:
            self._streak += 1
            limit = self.on_after if value else self.off_after
            if self._streak >= limit:
                self.state, self._streak = value, 0
        return self.state
