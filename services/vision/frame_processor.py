import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from services.config.workout_config import POSE_CONNECTIONS
from services.scoring.form_scorer import get_form_score


_landmarker = None

def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        model_path = os.path.join(os.getcwd(), "ml_models", "pose_landmarker_full.task")
        base_option = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_option,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_segmentation_masks=False
        )
        _landmarker = vision.PoseLandmarker.create_from_options(options)
    return _landmarker


def _draw_skeleton(img, landmarks):
    h, w = img.shape[:2]
    for start_idx, end_idx in POSE_CONNECTIONS:
        p1 = landmarks[start_idx]
        p2 = landmarks[end_idx]
        if p1.visibility > 0.7 and p2.visibility > 0.7:
            cv2.line(img,
                (int(p1.x * w), int(p1.y * h)),
                (int(p2.x * w), int(p2.y * h)),
                (0, 255, 0), 8)
    for lm in landmarks:
        if lm.visibility > 0.7:
            cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 8, (255, 0, 0), -1)


def _draw_form_score(img, score):
    h, w = img.shape[:2]
    color = (0, 255, 0) if score >= 80 else (0, 165, 255) if score >= 50 else (0, 0, 255)
    cv2.rectangle(img, (w - 250, 10), (w - 10, 65), (0, 0, 0), -1)
    cv2.putText(img, f"FORM: {score}/100", (w - 245, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def _draw_overlays(img, metrics, ex_type):
    h, _ = img.shape[:2]
    if ex_type == "Squats":
        cv2.putText(img, f"DEPTH: {metrics.get('depth_status', '')}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif ex_type == "Push-ups":
        cv2.putText(img, f"BODY: {metrics.get('body_alignment', '')} | HIP: {metrics.get('hip_status', '')}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif ex_type == "Biceps Curls (Dumbbell)":
        cv2.putText(img, f"SWING: {metrics.get('swing_status', '')}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif ex_type == "Shoulder Press":
        cv2.putText(img, f"EXT: {metrics.get('extension_status', '')} | BACK: {metrics.get('back_arch_status', '')}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif ex_type == "Lunges":
        cv2.putText(img, f"BALANCE: {metrics.get('balance_status', '')}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def process_frame(frame_bgr, exercise_type, detector):
    """Process a single frame and return (annotated_frame, metrics)."""
    try:
        landmarker = _get_landmarker()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            _draw_skeleton(frame_bgr, landmarks)

            if detector:
                metrics = detector.process(landmarks)
                metrics["pose_detected"] = True

                form_score, form_feedbacks = get_form_score(exercise_type, landmarks)
                metrics["form_score"] = form_score
                metrics["form_feedbacks"] = form_feedbacks

                _draw_overlays(frame_bgr, metrics, exercise_type)
                _draw_form_score(frame_bgr, form_score)
                return frame_bgr, metrics
        else:
            cv2.putText(frame_bgr, "NO POSE DETECTED", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_bgr, "PLEASE FACE THE CAMERA", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame_bgr, {"pose_detected": False}

    except Exception as e:
        cv2.putText(frame_bgr, f"Error: {str(e)[:40]}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame_bgr, {}

    return frame_bgr, {}
