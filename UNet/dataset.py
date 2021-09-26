#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# Custom Dataset Class
class ANI717Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, self.annotations.iloc[index,0])), cv2.COLOR_BGR2RGB)
        mask = np.array(cv2.imread(os.path.join(self.root_dir, self.annotations.iloc[index,1]), 0), dtype=np.float32)
        mask[mask < 200.0] = 0.0
        mask[mask != 0.0] = 1.0
        
        if self.transforms:
            augmentations = self.transforms(image=image, mask=mask)
            image, mask = augmentations["image"], augmentations["mask"]

        return image, mask





# # Test Block
# import config
# import torchvision
# from torch.utils.data import DataLoader
# from utils import save_sample_image

# dataset = ANI717Dataset('../Dataset/val.csv', '../Dataset/cityscapes/val', transforms=config.val_transforms)
# loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
# sample_inputs, sample_masks = next(iter(loader))
# img_grid_inputs = torchvision.utils.make_grid(sample_inputs[:8], nrow=4, normalize=True)
# img_grid_masks = torchvision.utils.make_grid(sample_masks.unsqueeze(1)[:8], nrow=4, normalize=True)
# save_sample_image(img_grid_inputs, img_grid_masks, img_grid_masks, 100)