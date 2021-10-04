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
    def __init__(self, videos_dir, transform=None):
        self.videos_dir = videos_dir
        self.transform = transform
        self.videos = os.listdir(self.videos_dir)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_path = os.path.join(self.videos_dir, self.videos[index])
        # frames,v_len = utils.
        frames, v_len = utils.get_frames_keypoints(video_path, n_frames=45)
        label = self.videos[index][self.videos[index].find(
            "_")+1:self.videos[index].find(".")]
        if self.transform is not None:
            pass

        return {
            "frame_keypoints": frames,
            "label": label,
            # "v_len":v_len
        }
