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
import utils


class PoseDataset(Dataset):
    def __init__(self, videos_dir, videos_name_list, transform=None, n_frames=45):
        self.videos_dir = videos_dir
        self.transform = transform
        self.n_frames = n_frames
        self.videos_name = videos_name_list

    def __len__(self):
        return len(self.videos_name)

    def __getitem__(self, index):
        video_path = os.path.join(self.videos_dir, self.videos_name[index])
        frames, v_len = utils.get_frames_keypoints(
            video_path, n_frames=self.n_frames)
        label = self.videos_name[index][self.videos_name[index].find(
            "_")+1:self.videos_name[index].find(".")]
        if self.transform is not None:
            pass

        return {
            "frame_keypoints": frames,
            "label": label,
            # "v_len":v_len
        }
