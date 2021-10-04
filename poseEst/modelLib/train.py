from numpy.lib import utils
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
import config
import utils
from dataset import PoseDataset


def fit(model, dataloader, data, optimzer, loss_fn):
    pass


def validate(model, dataloader, data, loss_fn):
    pass


if __name__ == "__main__":
    images = os.listdir(config.root_path)
    # one problem that can be faced here is all videos may not split
    # well as few types of them may not be found in test set

    train_images_names, val_images_names = utils.train_test_split(
        images, config.SPLIT)
    print(train_images_names)
    print('\n------------Creating Dataset Objects------------\n')
    train_data = PoseDataset(videos_dir=config.root_path,
                             videos_name_list=train_images_names)
    val_data = PoseDataset(videos_dir=config.root_path,
                           videos_name_list=val_images_names)
    print('\n------------Completed Dataset Creation-----------\n')

    print('\n----------------Creating DataLoaders-------------\n')

    train_dataloader = DataLoader(
        train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(
        val_data, batch_size=config.BATCH_SIZE, shuffle=False)
    print('\n----------------Dataloaders Done-----------------\n')
    
    # model = model.to(config.device)
    # optimizer = optim.Adam(model.parameters(),lr=config.LR)
    # loss_fn = nn.CrossEntropyLoss()

    print(train_data.__getitem__(0))