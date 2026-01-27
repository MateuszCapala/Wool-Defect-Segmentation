import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from dataset import WoolDataset, get_transforms
from torch.utils.data import DataLoader
from utils import post_process_mask

def create_qualitative_panel():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = getattr(smp, config['model']['architecture'])(
        encoder_name=config['model']['encoder'],
        in_channels=config['model']['in_channels'],
        classes=1
    ).to(config['train']['device'])
    
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    dataset = WoolDataset(config['data']['train_dir'], is_train=True, transform=get_transforms(config['data']['img_size'], is_train=False))
    loader = DataLoader(dataset, batch_size=1, shuffle=True) 

    os.makedirs("results_vis", exist_ok=True)
    num_samples = 5 

    

    with torch.no_grad():
        for i, (image, mask, uuid) in enumerate(loader):
            if i >= num_samples:
                break
            
            image_gpu = image.to(config['train']['device'])
            pred = torch.sigmoid(model(image_gpu))
            pred = (pred > 0.5).cpu().numpy().squeeze()
          
            pred_processed = post_process_mask(pred, min_area=100)

            raw_img = image.squeeze().numpy()
            gt_mask = mask.squeeze().numpy()

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            ax[0].imshow(raw_img, cmap='gray')
            ax[0].set_title(f"Raw X-Ray\n{uuid[0]}")
            ax[0].axis('off')

            ax[1].imshow(gt_mask, cmap='gray')
            ax[1].set_title("Ground Truth Mask")
            ax[1].axis('off')

            ax[2].imshow(pred_processed, cmap='magma')
            ax[2].set_title("Model Prediction")
            ax[2].axis('off')

            plt.tight_layout()
            plt.savefig(f"results_vis/panel_{uuid[0]}.png")
            plt.close()
            print(f"Zapisano panel dla: {uuid[0]}")

if __name__ == "__main__":
    create_qualitative_panel()