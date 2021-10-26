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
from . import config, utils
from .dataset import PoseDataset
from .models.PoseClassifier import PoseClassifier


def fit(model, dataloader, data, optimizer, loss_fn):
    print('\n-------------------Training--------------------\n')
    model.train()
    train_running_loss = 0.0
    counter = 0

    # num of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        keypoints, labels = data["keypoints"], data["label"]
        optimizer.zero_grad()
        if len(keypoints.shape) != 5:
            keypoints = keypoints.unsqueeze(2)

        outputs = model(keypoints)
        outputs = torch.transpose(outputs, 1, 2)
        labels = labels.to(torch.long)
        loss = loss_fn(outputs, labels)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss/counter
    return train_loss


def validate(model, dataloader, data, loss_fn):
    print('\n------------------Validating--------------------\n')
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    # num of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            keypoints, labels = data["keypoints"], data["label"]
            if len(keypoints.shape) != 5:
                keypoints = keypoints.unsqueeze(2)
            outputs = model(keypoints)
            # outputs is of format [batch_size,num_of_frames,num_of_classes]
            # we need to convert it to [batch_size,num_of_classes,num_of_frames]
            outputs = torch.transpose(outputs, 1, 2)
            labels = labels.to(torch.long)
            # print(outputs.shape,labels.shape)
            loss = loss_fn(outputs, labels)
            valid_running_loss += loss.item()
    valid_loss = valid_running_loss/counter
    return valid_loss


if __name__ == "__main__":
    images = os.listdir(config.root_path)
    # one problem that can be faced here is all videos may not split
    # well as few types of them may not be found in test set

    train_images_names, val_images_names = utils.train_test_split(
        images, config.SPLIT)
    # print(train_images_names)
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

    model = PoseClassifier().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(config.DEVICE)

    train_loss = []
    val_loss = []
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1} of {config.EPOCHS}")
        train_epoch_loss = fit(model, train_dataloader,
                               train_data, optimizer, loss_fn)
        val_epoch_loss = validate(model, val_dataloader, val_data, loss_fn)

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {val_epoch_loss:.4f}')

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", label='train loss')
    plt.plot(val_loss, color="red", label='validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.savefig(f"../input/loss.png")
    plt.show()
    torch.save({
        'epoch': config.EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_fn,
    }, "./model.pth")

    print("\n---------DONE TRAINING----------\n")
