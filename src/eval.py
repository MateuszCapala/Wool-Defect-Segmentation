import torch
import yaml
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import WoolDataset, get_transforms
from torchmetrics import F1Score, JaccardIndex, Precision, Recall
from tqdm import tqdm

def evaluate():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = getattr(smp, config['model']['architecture'])(
        encoder_name=config['model']['encoder'],
        in_channels=config['model']['in_channels'],
        classes=1
    ).to(config['train']['device'])
    
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    val_ds = WoolDataset(
        config['data']['train_dir'], 
        is_train=True, 
        transform=get_transforms(config['data']['img_size'], is_train=False)
    )
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'], shuffle=False)

    device = config['train']['device']
    metrics = {
        "F1": F1Score(task="binary").to(device),
        "IoU": JaccardIndex(task="binary").to(device),
        "Precision": Precision(task="binary").to(device),
        "Recall": Recall(task="binary").to(device)
    }

    print(f"Ewaluacja modelu na {len(val_ds)} próbkach")

    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device).int()
    
            outputs = model(images)
            preds = torch.sigmoid(outputs).squeeze(1)
            
            for m in metrics.values():
                m.update(preds, masks.squeeze(1))

    for name, m in metrics.items():
        print(f"{name}: {m.compute():.4f}")

if __name__ == "__main__":
    evaluate()