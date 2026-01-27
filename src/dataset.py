import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WoolDataset(Dataset):
    def __init__(self, data_dir, is_train=True, img_size=(448, 448), transform=None):
        self.is_train = is_train
        self.img_size = img_size
        self.transform = transform
        
        self.raw_path = os.path.join(data_dir, "raw")
        self.files = [f.replace("_raw.npy", "") for f in os.listdir(self.raw_path) if f.endswith(".npy")]
        
        if is_train:
            self.mask_path = os.path.join(data_dir, "masks")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        uuid = self.files[idx]
        
        raw_data = np.load(os.path.join(self.raw_path, f"{uuid}_raw.npy")).astype(np.float32)

        mean = raw_data.mean()
        std = raw_data.std()
        raw_data = (raw_data - mean) / (std + 1e-6)
        
        raw_min, raw_max = raw_data.min(), raw_data.max()
        if raw_max - raw_min > 0:
            raw_data = (raw_data - raw_min) / (raw_max - raw_min)

        if self.is_train:
            mask = cv2.imread(os.path.join(self.mask_path, f"{uuid}_mask.png"), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32) 
        else:
            mask = np.zeros(self.img_size, dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=raw_data, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(raw_data, self.img_size)
            mask = cv2.resize(mask, self.img_size)
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask.float(), uuid

def get_transforms(img_size, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            ToTensorV2()
        ])