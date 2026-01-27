import os
import yaml
import torch
import cv2
import numpy as np
import zipfile
import segmentation_models_pytorch as smp
from dataset import WoolDataset, get_transforms
from torch.utils.data import DataLoader
from utils import post_process_mask

def run_inference():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = getattr(smp, config['model']['architecture'])(
        encoder_name=config['model']['encoder'],
        in_channels=config['model']['in_channels'],
        classes=1
    ).to(config['train']['device'])
    
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_ds = WoolDataset(config['data']['test_dir'], is_train=False, transform=get_transforms(config['data']['img_size'], is_train=False))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    temp_folder = "submission"
    os.makedirs(temp_folder, exist_ok=True)

    print(f"--- Start inferencji dla {len(test_loader)} plików ---")

    with torch.no_grad():
        for images, _, uuids in test_loader:
            images = images.to(config['train']['device'])
            preds = torch.sigmoid(model(images))
            preds = (preds > 0.5).cpu().numpy().squeeze()
            
            mask_uint8 = (preds * 255).astype(np.uint8)
            
            processed_mask = post_process_mask(mask_uint8, min_area=100)
            
            mask_final = cv2.resize(processed_mask, (448, 448), interpolation=cv2.INTER_NEAREST)
            
            cv2.imwrite(os.path.join(temp_folder, f"{uuids[0]}_mask.png"), mask_final)

    zip_name = "masks.zip"
    print(f"--- Pakowanie masek do {zip_name} ---")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_folder):
            for file in files:
                if file.endswith(".png"):
                    full_path = os.path.join(root, file)
                    zipf.write(full_path, arcname=file)

    print(f"--- Gotowe! Plik {zip_name} jest gotowy do wysyłki. ---")

if __name__ == "__main__":
    run_inference()