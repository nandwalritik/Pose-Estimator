from ..modelLib.models import PoseDetector
import cv2
import imutils
from ..modelLib.utils.utils import showImage
import numpy as np


PD = PoseDetector(True, 2, False, 0.5)
img = cv2.imread('../demoImages/demo3.jpg')

image_height, image_width, _ = img.shape
img = imutils.resize(img, width=720)
showImage(img, "Input Image")

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

# Draw pose landmarks on the image.
PD.mp_drawing.draw_landmarks(
    annotated_image,
    results.pose_landmarks,
    PD.mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=PD.mp_drawing_styles.get_default_pose_landmarks_style())
showImage(annotated_image, "Annotated Image")
