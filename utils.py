import torch
import torchvision
from dataset import BraTSDataset
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.distance import cdist

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def hausdorff_distance(y_pred, y_true, spacing=[1., 1., 1.], percent=0.95, num_class=1, decimal=4):
    """
    calculate the 95% (by default) hausdorff distance between the contour of prediction and ground truth
    """
    res = []

    for i in range(num_class):
        target = y_true[i]
        pred = y_pred[i]

        if target.sum() and pred.sum():
            a_pts = np.where(target)
            b_pts = np.where(pred)
            a_pts = np.array(a_pts).T * np.array(spacing)
            b_pts = np.array(b_pts).T * np.array(spacing)

            dists = cdist(a_pts, b_pts)
            a = np.min(dists, 1)
            b = np.min(dists, 0)
            a.sort()
            b.sort()

            a_max = a[int(percent * len(a)) - 1]
            b_max = b[int(percent * len(b)) - 1]

            res.append(round(max(a_max, b_max), decimal))
        else:
            res.append(None)
    return res

def get_loaders(
    train_dir_flair,
    train_dir_t1,
    train_dir_t2,
    train_dir_t1ce,
    train_maskdir,
    val_dir_flair,
    val_dir_t1,
    val_dir_t2,
    val_dir_t1ce,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = BraTSDataset(
        flair_dir = train_dir_flair, 
        t1_dir = train_dir_t1, 
        t2_dir = train_dir_t2, 
        t1ce_dir = train_dir_t1ce,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = BraTSDataset(
        flair_dir = val_dir_flair, 
        t1_dir = val_dir_t1, 
        t2_dir = val_dir_t2, 
        t1ce_dir = val_dir_t1ce,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    hd95 = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            hd95 = hausdorff_distance(y_pred = preds, y_true = y)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    print(f"HD95 score: {hd95}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1).float(), f"{folder}/seg_{idx}.png")

    model.train()