import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import imageio


class PoseDataset(Dataset):
    def __init__(self, videos_dir, videos_name_list, transform=None, n_frames=70):
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
        # print('\n------------PrintPath------------\n')
        # print(video_path)
        keypoints, v_len = get_frames_keypoints(
            video_path, n_frames=self.n_frames)
        label = self.videos_name[index][self.videos_name[index].find(
            "_")+1:self.videos_name[index].find(".")]
        label = int(self.le.transform([label])[0])
        # print(label)
        if self.transform is not None:
            pass

        # Here frames is a dict containing frame and posekeypoints
        # {"frame": frame, "pose_keypoints": lmkList}
        targetTensor = torch.ones(45)*label

        return {
            "keypoints": torch.tensor(keypoints, dtype=torch.float).to(config.DEVICE),
            "label": torch.tensor(targetTensor, dtype=torch.float).to(config.DEVICE),
            # "v_len":v_len
        }


# Uncomment below to check visualizations
# if __name__ == "__main__":
#     root_path = "./datasets/YogaVidCollected/Yoga_Vid_Collected"
#     images = os.listdir(root_path)
#     temp = PoseDataset(root_path, images)
#     data = temp.__getitem__(5)
#     print(data["label"])
#     print(np.array(data['keypoints'].detach().cpu()).shape)
#     print(data['keypoints'][0])
#     # plt.scatter(temp[:,0],temp[:,1])
#     # plt.show()
#     frameList = []
#     for i in range(len(data["keypoints"])):
#         img = np.ones((700, 1000, 3))
#         temp = np.array(data["keypoints"][i].detach().cpu())
#         for x in temp:
#             # print(x)
#             # substracted 200 pixels to adjust pose inside frame
#             cv2.circle(img, (int(x[0])-200, int(x[1])), 3, (0, 0, 0), 3)
#         cv2.imshow("Yoga Asana", img)
#         cv2.waitKey(200)
#         # frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         frameList.append(img)
#     imageio.mimsave('./SamplePose.gif', frameList, fps=10)
    
