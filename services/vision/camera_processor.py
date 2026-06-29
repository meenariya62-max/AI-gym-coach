import base64
import io
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector
from services.config.workout_config import POSE_CONNECTIONS
from services.scoring.form_scorer import get_form_score


class CameraProcessor:
    def __init__(self):
        model_path = os.path.join(os.getcwd(), "ml_models", "pose_landmarker_full.task")
        base_option = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_option,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
        }

    def process_frame(self, frame_bytes: bytes, exercise_type: str) -> dict:
        # bytes → numpy image
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {"pose_detected": False}

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return {"pose_detected": False}

        landmarks = result.pose_landmarks[0]
        detector = self._detectors.get(exercise_type)
        if not detector:
            return {"pose_detected": False}

        metrics = detector.process(landmarks)
        metrics["pose_detected"] = True

        form_score, form_feedbacks = get_form_score(exercise_type, landmarks)
        metrics["form_score"] = form_score
        metrics["form_feedbacks"] = form_feedbacks

        # Draw skeleton
        h, w = img_bgr.shape[:2]
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                p1, p2 = landmarks[start_idx], landmarks[end_idx]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    cv2.line(img_bgr,
                             (int(p1.x * w), int(p1.y * h)),
                             (int(p2.x * w), int(p2.y * h)),
                             (0, 255, 0), 4)
        for lm in landmarks:
            if lm.visibility > 0.5:
                cv2.circle(img_bgr, (int(lm.x * w), int(lm.y * h)), 6, (255, 0, 0), -1)

        # Form score overlay
        color = (0,255,0) if form_score >= 80 else ((0,165,255) if form_score >= 50 else (0,0,255))
        cv2.rectangle(img_bgr, (w-200, 10), (w-10, 55), (0,0,0), -1)
        cv2.putText(img_bgr, f"FORM: {form_score}/100", (w-195, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Encode to base64
        _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        metrics["annotated_frame"] = annotated_b64

        return metrics
