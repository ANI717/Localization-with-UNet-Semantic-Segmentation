#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Hyperparameters
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
PIN_MEMORY = True

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT = "checkpoint/model.pth.tar"
RESULTS = 'results'


# Transformations
train_transforms = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
    A.Rotate(limit=35, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])