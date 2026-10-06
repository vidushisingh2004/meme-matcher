# Meme Matcher

Point your webcam at your face and it shows you the meme whose expression
looks most like yours. Hold up both open hands and you get the
"wait, wait, wait!" meme.

## Setup (one time)

You need Python 3.9 - 3.12 and a webcam.

```bash
cd meme-matcher
python -m venv venv
# Windows:  venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

First launch needs internet: it downloads two small MediaPipe model files
(~4 MB face, ~8 MB hands) into `models/`, then reads your meme images and
writes `memes.json`. After that it works offline.

## Controls

| Key | Action |
|-----|--------|
| `c` | Calibrate: hold a relaxed neutral face, press `c`. Fixes people whose resting face looks "frowny" or "surprised" to the camera |
| `r` | Reset calibration |
| `d` | Debug panel: live match score for every meme |
| `s` | Save a snapshot of you + your meme into `snapshots/` |
| `q` / Esc | Quit |

Options: `--camera 1` (pick another webcam), `--no-hands` (turn off the
hands-up gesture), `--rebuild` (re-read the meme images into `memes.json`).

## How it works

1. **OpenCV** grabs webcam frames.
2. **MediaPipe Face Landmarker** turns each face into 52 "blendshape"
   scores (`jawOpen`, `eyeWideLeft`, `browInnerUp`, ...), each 0 to 1.
3. `build_library.py` runs the same model on each meme image to get its
   expression vector.
4. `matcher.py` compares your vector to every meme with cosine
   similarity, scaled by how intense your expression is, averaged over a
   few frames, and only switches memes after it wins several frames in a row.
5. **MediaPipe Hand Landmarker** spots two open hands for the gesture meme.

## Files

```
app.py             webcam loop + display
matcher.py         scoring, smoothing, switching (the tunable part)
build_library.py   meme images -> memes.json
gestures.py        open-hands detection
common.py          paths, model download, blendshape weights
memes_config.json  your memes + hand-tagged backup vectors
memes/             the meme images
```

## Tuning

- **Meme never triggers / triggers too easily:** edit `MIN_SCORE` and
  `INTENSITY_FRACTION` at the top of `matcher.py`.
- **One meme wins too often:** press `d`, watch the scores, then lower that
  meme's influence in `WEIGHTS` in `common.py` or fix its vector in
  `memes_config.json`.
- **A meme's face reads wrong** (e.g. a hand covers the mouth): in
  `memes_config.json` set its `mode` to `blend` (average of detected and
  hand-tagged) or `manual` (hand-tagged only), then `python app.py --rebuild`.
  `build_library.py` prints each meme's top features so you can sanity-check.

## Adding a meme

1. Drop the image in `memes/`.
2. Add an entry to `memes_config.json` (`mode: "auto"`, plus a
   `manual_vector` as a backup).
3. `python app.py --rebuild`

## Troubleshooting

- *Camera won't open:* close Zoom/Teams/etc., check OS camera permissions,
  or try `--camera 1`.
- *Model download fails:* download the files from the URLs in `common.py`
  and put them in `models/` with the same filenames.
- *Everything matches the wrong meme:* use `c` to calibrate, and make
  bigger expressions. Meme faces are extreme.
