import torch

LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 1
NUM_WORKERS = 2
root_path = "./datasets/YogaVidCollected/Yoga_Vid_Collected"

SPLIT = 0.2
