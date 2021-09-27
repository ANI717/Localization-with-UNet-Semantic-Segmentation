#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import os
import torch
import config
from pathlib import Path
from torchvision.utils import save_image


# Load Checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Save Checkpoint
def save_checkpoint(model, optimizer, filename='checkpoint/my_checkpoint.pth.tar'):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        }
    torch.save(checkpoint, filename)


# Save Sample Image
def save_sample_image(inputs, targets, predictions, epoch):
    Path(config.RESULTS).mkdir(parents=True, exist_ok=True)
    save_image(inputs, os.path.join(config.RESULTS, 'input.jpg'))
    save_image(targets, os.path.join(config.RESULTS, 'target.jpg'))
    save_image(predictions, os.path.join(config.RESULTS, f'prediction_epoch{epoch+1}.jpg'))


# Accuracy Check
def check_accuracy(loader, model, device="cpu"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / ((preds+y).sum()+1e-8) # py/(p+y)

    print(f"Accuracy: {num_correct/num_pixels*100:.2f}%")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()