import numpy as np

# Standard MediaPipe Pose landmark indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def _angle(a, b, c):
    """Angle at point b, formed by points a-b-c, in degrees."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _vertical_angle(a, b):
    """Angle of segment a-b from vertical axis, in degrees."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    vec = b - a
    vertical = np.array([0, 1])
    cos_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _score_squats(lm):
    score = 100
    feedbacks = []

    knee_angle = _angle(lm[LEFT_HIP], lm[LEFT_KNEE], lm[LEFT_ANKLE])
    hip_angle = _angle(lm[LEFT_SHOULDER], lm[LEFT_HIP], lm[LEFT_KNEE])
    back_angle = _vertical_angle(lm[LEFT_SHOULDER], lm[LEFT_HIP])

    # Knee angle — 70 se 130 ke beech theek hai
    if knee_angle > 150:
        feedbacks.append("⬇️ Bend your knees more")
        score -= 15
    elif knee_angle < 60:
        feedbacks.append("⚠️ Too deep, watch your knees")
        score -= 10

    # Back angle — 55 degree tak allow kiya (pehle 45 tha)
    if back_angle > 55:
        feedbacks.append("🔙 Keep your back straighter")
        score -= 20

    # Hip angle — 50 degree tak allow kiya (pehle 60 tha)
    if hip_angle < 50:
        feedbacks.append("↩️ Push your hips back more")
        score -= 10

    return score, feedbacks


def _score_pushups(lm):
    score = 100
    feedbacks = []

    elbow_angle = _angle(lm[LEFT_SHOULDER], lm[LEFT_ELBOW], lm[LEFT_WRIST])
    body_line_angle = _angle(lm[LEFT_SHOULDER], lm[LEFT_HIP], lm[LEFT_ANKLE])

    # Body line — 150 tak allow kiya (pehle 160 tha)
    if body_line_angle < 150:
        feedbacks.append("📏 Keep body in straight line, don't sag hips")
        score -= 20

    # Elbow — 175 tak allow kiya
    if elbow_angle > 175:
        feedbacks.append("⬇️ Lower yourself more for full range")
        score -= 10

    return score, feedbacks


def _score_biceps_curl(lm):
    score = 100
    feedbacks = []

    elbow_angle = _angle(lm[LEFT_SHOULDER], lm[LEFT_ELBOW], lm[LEFT_WRIST])
    shoulder_movement = _vertical_angle(lm[LEFT_HIP], lm[LEFT_SHOULDER])

    # Shoulder swing — 20 degree tak allow kiya (pehle 15 tha)
    if shoulder_movement > 20:
        feedbacks.append("🚫 Avoid swinging shoulders, keep upper arm still")
        score -= 20

    if elbow_angle < 30:
        feedbacks.append("✅ Great curl range!")
    elif elbow_angle > 175:
        feedbacks.append("💪 Don't fully lock out, keep slight tension")
        score -= 5

    return score, feedbacks


def _score_shoulder_press(lm):
    score = 100
    feedbacks = []

    elbow_angle = _angle(lm[LEFT_SHOULDER], lm[LEFT_ELBOW], lm[LEFT_WRIST])
    back_angle = _vertical_angle(lm[LEFT_SHOULDER], lm[LEFT_HIP])

    # Back arch — 25 degree tak allow kiya (pehle 20 tha)
    if back_angle > 25:
        feedbacks.append("🔙 Avoid arching your back during press")
        score -= 20

    # Arm extension — 140 tak allow kiya (pehle 150 tha)
    if elbow_angle < 140:
        feedbacks.append("🙌 Extend your arms fully overhead")
        score -= 10

    return score, feedbacks


def _score_lunges(lm):
    score = 100
    feedbacks = []

    front_knee_angle = _angle(lm[LEFT_HIP], lm[LEFT_KNEE], lm[LEFT_ANKLE])
    torso_angle = _vertical_angle(lm[LEFT_SHOULDER], lm[LEFT_HIP])

    # Knee — 65 tak allow kiya (pehle 70 tha)
    if front_knee_angle < 65:
        feedbacks.append("⚠️ Don't let front knee go too far past toes")
        score -= 15

    # Torso — 25 degree tak allow kiya (pehle 20 tha)
    if torso_angle > 25:
        feedbacks.append("🧍 Keep your torso upright")
        score -= 15

    return score, feedbacks


_SCORERS = {
    "Squats": _score_squats,
    "Push-ups": _score_pushups,
    "Biceps Curls (Dumbbell)": _score_biceps_curl,
    "Shoulder Press": _score_shoulder_press,
    "Lunges": _score_lunges,
}


def get_form_score(ex_type, landmarks):
    """
    Returns (score: int, feedbacks: list[str]) for the given exercise type
    and a list of MediaPipe pose landmarks.
    """
    scorer = _SCORERS.get(ex_type)
    if scorer is None:
        return 100, []

    try:
        score, feedbacks = scorer(landmarks)
    except (IndexError, AttributeError):
        return 0, ["⚠️ Pose not fully visible"]

    score = int(max(0, min(100, score)))
    return score, feedbacks