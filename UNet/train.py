#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from pathlib import Path

import config
from model import UNet
from dataset import ANI717Dataset
from utils import save_checkpoint, load_checkpoint, save_sample_image, check_accuracy


# Seed for Reproducibility
torch.manual_seed(0)


# Main Method
def main():
    
    # Load Data
    train_dataset = ANI717Dataset('../Dataset/train.csv', '../Dataset/cityscapes/train', transforms=config.train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_dataset = ANI717Dataset('../Dataset/val.csv', '../Dataset/cityscapes/val', transforms=config.val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    sample_inputs, sample_masks = next(iter(val_loader))
    
    # Initialize Model, Optimizer and Loss
    model = UNet(in_channels=3, out_channels=1).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    # Load Checkpoint
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT, model, optimizer, config.LEARNING_RATE)
    
    # # Test Block
    # sample_images, sample_masks = next(iter(train_loader))
    # print(sample_images.shape, sample_masks.shape)
    # import sys
    # sys.exit()
    
    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader)
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs = inputs.to(config.DEVICE)
            targets = targets.float().unsqueeze(1).to(config.DEVICE)
            
            # Forward
            with torch.cuda.amp.autocast():
                predictions = model(inputs)
                loss = criterion(predictions, targets)
            
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update tqdm loop
            loop.set_postfix(loss=loss.item())
        
        
        # Check accuracy
        check_accuracy(val_loader, model, config.DEVICE)
        
        # Save some samples
        with torch.no_grad():
            sample_predictions = torch.sigmoid(model(sample_inputs.to(config.DEVICE)))
            sample_predictions = (sample_predictions > 0.5).float()
            
            # # Take out (up to) 8 examples
            img_grid_inputs = torchvision.utils.make_grid(sample_inputs[:8], nrow=4, normalize=True)
            img_grid_masks = torchvision.utils.make_grid(sample_masks.unsqueeze(1)[:8], nrow=4, normalize=True)
            img_grid_predictions = torchvision.utils.make_grid(sample_predictions[:8], nrow=4, normalize=True)
            
            # Save Sample Generated Images
            save_sample_image(img_grid_inputs, img_grid_masks, img_grid_predictions, epoch)
        
        # Save Model
        if config.SAVE_MODEL:
            Path(config.CHECKPOINT.split('/')[0]).mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, config.CHECKPOINT)


if __name__ == "__main__":
    main()