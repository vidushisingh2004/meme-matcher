"""Tiny gesture helpers built on MediaPipe hand landmarks."""
import math

# (fingertip index, middle-joint index) for index, middle, ring, pinky
_FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def is_open_hand(landmarks):
    """True if all four fingers are extended (fingertip farther from the wrist
    than the finger's middle joint)."""
    wrist = landmarks[0]
    return all(_dist(landmarks[tip], wrist) > 1.15 * _dist(landmarks[mid], wrist)
               for tip, mid in _FINGERS)


def is_upright(landmarks):
    """Fingers point up: middle fingertip above its knuckle, knuckle above wrist."""
    return landmarks[12].y < landmarks[9].y < landmarks[0].y


def hands_up(hand_landmarks_list):
    """'Wait, wait, wait!' pose: two open hands visible at once."""
    return sum(1 for h in hand_landmarks_list if is_open_hand(h)) >= 2


def stop_sign(hand_landmarks_list):
    """'Talk to the hand' pose: exactly one open hand, held upright."""
    open_hands = [h for h in hand_landmarks_list if is_open_hand(h)]
    return len(open_hands) == 1 and is_upright(open_hands[0])


# gesture name (as used in memes_config.json) -> detector
GESTURES = {"hands_up": hands_up, "stop": stop_sign}
GESTURE_PRIORITY = ["hands_up", "stop"]   # first match wins
