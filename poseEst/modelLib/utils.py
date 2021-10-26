"""
    This file contains utility functions
"""

import cv2
import numpy as np
import os
from .models.PoseDetector import PoseDetector


"""-----------Show Image----------"""

WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)


def draw_line(image, p1, p2, color):
    cv2.line(image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)


def showImage(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)


def get_frames_keypoints(filename, n_frames=1):
    keypoints_arr = []
    PD = PoseDetector(False, 1, False, 0.5)
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(v_len)
    frame_list = np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    # print(frame_list)
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # print("Frame dims  "+str(frame.shape))
            keypoints = PD.detectJoints(frame)
            lmkList = []
            try:
                for id, pnt in enumerate(keypoints.pose_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(pnt.x*w), int(pnt.y*h)
                    if id in [0, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28]:
                        lmkList.append([cx, cy])
                    # frame = np.transpose(frame, (2, 0, 1))
                keypoints_arr.append(lmkList)
            except:
                print("No keypoints detected")

    v_cap.release()
    return keypoints_arr[:45], v_len


def store_frames(frames, path2store):
    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path2img = os.path.join(path2store, "frame"+str(i)+".jpg")
        cv2.imwrite(path2img, frame)


def train_test_split(videos_paths, split):
    total_len = len(videos_paths)
    val_len = int(split*total_len)
    train_len = total_len - val_len
    train_videos_path, val_videos_path = videos_paths[:
                                                      train_len], videos_paths[train_len:]
    return train_videos_path, val_videos_path
