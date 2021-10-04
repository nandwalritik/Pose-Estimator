from cv2 import batchDistance
import torch

LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 20
NUM_WORKERS = 2
# DATAPATH = pass
SPLIT = 0.2