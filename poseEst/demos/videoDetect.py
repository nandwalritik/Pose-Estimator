from ..modelLib.models.PoseDetector import PoseDetector
import cv2
import imutils
from ..modelLib.utils.utils import showImage
import numpy as np

PD = PoseDetector(True, 2, False, 0.5)

cap = cv2.VideoCapture(
    '/home/nandwalritik/poseEstimator/poseEst/TestSamples/inputs/testVideo1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (videoWidth, videoHeight)

out = cv2.VideoWriter(
    '/home/nandwalritik/poseEstimator/poseEst/TestSamples/outputs/testVideo1.mp4', fourcc, 20, size)
while(cap.isOpened()):
    ret, frame = cap.read()
    if(ret == True):
        results = PD.detectJoints(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # print(results.pose_landmarks)

        annotated_image = frame.copy()
        PD.mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            PD.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=PD.mp_drawing_styles.get_default_pose_landmarks_style())
    
        # cv2.imshow("frame",annotated_image)

        out.write(annotated_image)
        
        # Press q to stop window
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
