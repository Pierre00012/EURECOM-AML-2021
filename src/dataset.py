
import os
import json
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RefugeDataset(Dataset):

    def __init__(self, root_dir, split="train", transform=None):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.split_dir = os.path.join(root_dir, split)

        # Charger les annotations
        json_path = os.path.join(
            self.split_dir, "index.json"
        )

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.df = pd.json_normalize(list(data.values()))

        self.image_dir = os.path.join(
            self.split_dir, "images"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_name = row["ImgName"]
        label = int(row["Label"])

        image_path = os.path.join(
            self.image_dir, image_name
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def load_data(root_dir, batch_size=16):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = RefugeDataset(
        root_dir, "train", transform
    )

    val_dataset = RefugeDataset(
        root_dir, "val", transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader
