"""
    This file contains utility functions
"""
import cv2
"""-----------Show Image----------"""
def showImage(image,title):
    cv2.imshow(title,image)
    cv2.waitKey(0)
    