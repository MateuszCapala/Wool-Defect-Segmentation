import yaml
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
from dataset import WoolDataset, get_transforms
from torchmetrics import F1Score
import datetime
import numpy as np
import os 

import random


def train():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"Wool-UnetPP-ResNet50-LR{config['train']['lr']}-{now}"

    wandb.init(
        project=config['project_name'], 
        config=config,
        name=run_name,    
        notes="Pierwszy udany trening z F1=0.95" 
    )

    train_ds = WoolDataset(config['data']['train_dir'], is_train=True, transform=get_transforms(config['data']['img_size']))
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4)

    model = getattr(smp, config['model']['architecture'])(
        encoder_name=config['model']['encoder'],
        encoder_weights=config['model']['weights'],
        in_channels=config['model']['in_channels'],
        classes=1
    ).to(config['train']['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.1)
    best_f1 = 0
    patience_counter = 0
    early_stop_patience = 20
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()

    criterion = lambda pred, target: dice_loss(pred, target) + (0.5 * bce_loss(pred, target))
    scaler = torch.amp.GradScaler('cuda')
    f1_metric = F1Score(task="binary").to(config['train']['device'])

    model.train()
    for epoch in range(config['train']['epochs']):
        epoch_loss = 0
        for images, masks, _ in train_loader:
            images, masks = images.to(config['train']['device']), masks.to(config['train']['device'])
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            f1_metric.update(torch.sigmoid(outputs).squeeze(1), masks.int())

        avg_f1 = f1_metric.compute()
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
            print(f"--- Nowy rekord F1: {best_f1:.4f} - Zapisano model ---")
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"--- BRAK POPRAWY PRZEZ {early_stop_patience} EPOK. KOŃCZĘ. ---")
            break
        wandb.log({"loss": epoch_loss/len(train_loader), "f1_score": avg_f1})
        print(f"Epoch {epoch}: Loss {epoch_loss/len(train_loader):.4f}, F1: {avg_f1:.4f}")
        f1_metric.reset()

    torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    train()