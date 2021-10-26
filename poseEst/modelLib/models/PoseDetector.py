import mediapipe as mp
from mediapipe.python.solutions.pose import Pose
import cv2

class PoseDetector:
    def __init__(self, static_image_mode: bool, model_complexity: int, enable_segmentation: bool, min_detection_confidence: float) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.mediapipe.python.solutions.pose

        self.poseDetector = Pose(static_image_mode=static_image_mode,
                                 model_complexity=model_complexity,
                                 enable_segmentation=enable_segmentation,
                                 min_detection_confidence=0.2)

    def detectJoints(self, image):
        joints = self.poseDetector.process(image)
        return joints


