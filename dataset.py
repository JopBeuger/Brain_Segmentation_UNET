import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class BraTSDataset(Dataset):
    def __init__(self, flair_dir, t1_dir, t2_dir, t1ce_dir, mask_dir, transform=None):
        self.flair_dir = flair_dir
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.t1ce_dir = t1ce_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        flair_path = os.path.join(self.flair_dir, self.images[index])
        t1_path = os.path.join(self.t1_dir, self.images[index])
        t2_path = os.path.join(self.t2_dir, self.images[index])
        t1ce_path = os.path.join(self.t1ce_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        flair_image = np.expand_dims(np.array(Image.open(flair_path)), 0)
        t1_image = np.expand_dims(np.array(Image.open(t1_path)),0)
        t1ce_image = np.expand_dims(np.array(Image.open(t1ce_path)),0)
        t2_image = np.expand_dims(np.array(Image.open(t2_path)),0)
        image = np.concatenate((flair_image, t1_image, t1ce_image, t2_image), 0)
        image = np.moveaxis(image, 0, -1)
        mask = np.array(Image.open(mask_path))
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

