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
                                 min_detection_confidence=min_detection_confidence)

    def detectJoints(self, image):
        joints = self.poseDetector.process(image)
        return joints


PD = PoseDetector(True, 2, False, 0.5)

"""
#--------------------For Image----------------#
img = cv2.imread('../datasets/demoImages/demo3.jpg')

image_height, image_width, _ = img.shape
img = imutils.resize(img,width=720)
print(img.shape)
cv2.imshow('img',img)
cv2.waitKey(0)

results = PD.detectJoints(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print(results.pose_landmarks)
annotated_image = img.copy()
# Draw segmentation on the image.
# To improve segmentation around boundaries, consider applying a joint
# bilateral filter to "results.segmentation_mask" with "image".
condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
bg_image = np.zeros(img.shape, dtype=np.uint8)
bg_image[:] = (192, 192, 192)
annotated_image = np.where(condition, annotated_image, bg_image)
cv2.imshow('img',annotated_image)
cv2.waitKey(0)
# Draw pose landmarks on the image.
PD.mp_drawing.draw_landmarks(
    annotated_image,
    results.pose_landmarks,
    PD.mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=PD.mp_drawing_styles.get_default_pose_landmarks_style())
cv2.imwrite('/tmp/annotated_image' + str(1) + '.png', annotated_image)
cv2.imshow('img',annotated_image)
cv2.waitKey(0)
"""

#------------------------For Video-------------------------#
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         # If loading a video, use 'break' instead of 'continue'.
#         continue

#     # Flip the image horizontally for a later selfie-view display, and convert
#     # the BGR image to RGB.
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = PD.poseDetector.process(image)

#     # Draw the pose annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     PD.mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         PD.mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=PD.mp_drawing_styles.get_default_pose_landmarks_style())
#     cv2.imshow('MediaPipe Pose', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#         break
# cap.release()
