import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 60  # 240 originally
IMAGE_WIDTH = 60  # 240 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR_FLAIR = "/Users/jopbeuger/Documents/JADS/YEAR_2/Thesis/data/train/flair/"
TRAIN_IMG_DIR_T1 = "/Users/jopbeuger/Documents/JADS/YEAR_2/Thesis/data/train/t1/"
TRAIN_IMG_DIR_T2 = "/Users/jopbeuger/Documents/JADS/YEAR_2/Thesis/data/train/t2/"
TRAIN_IMG_DIR_T1CE = "/Users/jopbeuger/Documents/JADS/YEAR_2/Thesis/data/train/t1ce/"
TRAIN_MASK_DIR = "/Users/jopbeuger/Documents/JADS/YEAR_2/Thesis/data/train/seg/"
VAL_IMG_DIR_FLAIR = "/Users/jopbeuger/Documents/JADS/YEAR_2/Thesis/data/test/flair/"
VAL_IMG_DIR_T1 = "/Users/jopbeuger/Documents/JADS/YEAR_2/Thesis/data/test/t1/"
VAL_IMG_DIR_T2 = "/Users/jopbeuger/Documents/JADS/YEAR_2/Thesis/data/test/t2/"
VAL_IMG_DIR_T1CE = "/Users/jopbeuger/Documents/JADS/YEAR_2/Thesis/data/test/t1ce/"
VAL_MASK_DIR = "/Users/jopbeuger/Documents/JADS/YEAR_2/Thesis/data/test/seg/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean= [0.0, 0.0, 0.0, 0.0],
                std= [1.0, 1.0, 1.0, 1.0],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean= [0.0, 0.0, 0.0, 0.0],
                std= [1.0, 1.0, 1.0, 1.0],
                max_pixel_value= 1,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=4, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR_FLAIR,
        TRAIN_IMG_DIR_T1,
        TRAIN_IMG_DIR_T2,
        TRAIN_IMG_DIR_T1CE,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR_FLAIR,
        VAL_IMG_DIR_T1,
        VAL_IMG_DIR_T2,
        VAL_IMG_DIR_T1CE,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images", device=DEVICE
        )


if __name__ == "__main__":
    main()