import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
from .utils import get_frames_keypoints
from sklearn.preprocessing import LabelEncoder
from . import config

class PoseDataset(Dataset):
    def __init__(self, videos_dir, videos_name_list, transform=None, n_frames=44):
        self.videos_dir = videos_dir
        self.transform = transform
        self.n_frames = n_frames
        self.videos_name = videos_name_list
        self.le = LabelEncoder()
        self.classes = ["Bhujangasana", "Padmasana",
                        "Shavasana", "Tadasana", "Trikonasana", "Vrikshasana"]
        self.le.fit(self.classes)
        

    def __len__(self):
        return len(self.videos_name)

    def __getitem__(self, index):
        video_path = os.path.join(self.videos_dir, self.videos_name[index])
        keypoints, v_len = get_frames_keypoints(
            video_path, n_frames=self.n_frames)
        label = self.videos_name[index][self.videos_name[index].find(
            "_")+1:self.videos_name[index].find(".")]
        if self.transform is not None:
            pass

        # Here frames is a dict containing frame and posekeypoints
        # {"frame": frame, "pose_keypoints": lmkList}

        return {
            "keypoints": torch.tensor(keypoints,dtype=torch.float).to(config.DEVICE),
            "label": torch.tensor(self.le.transform([label]),dtype=torch.float).to(config.DEVICE),
            # "v_len":v_len
        }


# Uncomment below to check visualizations
# if __name__ == "__main__":
#     root_path = "./datasets/YogaVidCollected/Yoga_Vid_Collected"
#     images = os.listdir(root_path)
#     temp = PoseDataset(root_path, images)
#     data = temp.__getitem__(44)
#     print(data["label"])
#     # print(np.array(data['keypoints']).shape)
#     # print(data['keypoints'][0])
#     # plt.scatter(temp[:,0],temp[:,1])
#     # plt.show()
#     for i in range(len(data["keypoints"])):
#         img = np.ones((768, 1366, 3))
#         temp = np.array(data["keypoints"][i])
#         for x in temp:
#             # print(x)
#             cv2.circle(img, (x[0], x[1]), 2, (255, 255, 0), 2)
#         cv2.imshow(str(data["label"]), img)
#         cv2.waitKey(90)
