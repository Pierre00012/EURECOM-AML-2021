
# ============================================================
# model.py
# SegFormer-B0 + MLP pour la detection du glaucome
# Kaggle GPU compatible
#
# Label:
# 0 = Pas de glaucome
# 1 = Glaucome
#
# Variables tabulaires:
# Fovea_X, Fovea_Y
# ============================================================

import os
import json
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import SegformerModel




# ============================================================
# 1. DATASET
# ============================================================

class RefugeDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        tabular_mean=None,
        tabular_std=None
    ):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.split_dir = os.path.join(root_dir, split)

        json_path = os.path.join(
            self.split_dir, "index.json"
        )

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.df = pd.json_normalize(list(data.values()))

        self.image_dir = os.path.join(
            self.split_dir, "images"
        )

        # Seulement les variables utiles
        self.tabular_columns = [
            "Fovea_X",
            "Fovea_Y"
        ]

        self.df[self.tabular_columns] = (
            self.df[self.tabular_columns].astype(float)
        )

        # Statistiques calculees sur le train
        if tabular_mean is None:
            tabular_mean = self.df[
                self.tabular_columns
            ].mean().values

        if tabular_std is None:
            tabular_std = self.df[
                self.tabular_columns
            ].std().values

        tabular_std = np.where(
            np.array(tabular_std) == 0,
            1.0,
            tabular_std
        )

        self.tabular_mean = np.asarray(
            tabular_mean, dtype=np.float32
        )

        self.tabular_std = np.asarray(
            tabular_std, dtype=np.float32
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # Image
        image_name = row["ImgName"]

        image_path = os.path.join(
            self.image_dir, image_name
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Donnees tabulaires
        tabular = row[
            self.tabular_columns
        ].values.astype(np.float32)

        tabular = (
            tabular - self.tabular_mean
        ) / self.tabular_std

        tabular = torch.tensor(
            tabular, dtype=torch.float32
        )

        # Label:
        # 0 = Pas de glaucome
        # 1 = Glaucome

        label = int(row["Label"])

        label = torch.tensor(
            label, dtype=torch.long
        )

        return image, tabular, label


# ============================================================
# 2. MODELE MULTIMODAL
# ============================================================

class SegFormerGlaucoma(nn.Module):

    def __init__(
        self,
        pretrained=True,
        num_classes=2,
        dropout=0.3
    ):

        super().__init__()

        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"

        # Backbone SegFormer-B0
        if pretrained:
            self.image_encoder = SegformerModel.from_pretrained(
                "nvidia/mit-b0"
            )
        else:
            from transformers import SegformerConfig

            config = SegformerConfig()
            self.image_encoder = SegformerModel(config)

        # Dimension des features de SegFormer-B0
        image_dim = self.image_encoder.config.hidden_sizes[-1]

        # Branche tabulaire
        self.tabular_encoder = nn.Sequential(

            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Fusion image + tabulaire
        fusion_dim = image_dim + 64

        self.classifier = nn.Sequential(

            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

    def forward(self, image, tabular):

        # Branche image
        outputs = self.image_encoder(
            pixel_values=image
        )

        features = outputs.last_hidden_state

        # Global Average Pooling
        image_features = features.mean(
            dim=(2, 3)
        )

        # Branche tabulaire
        tabular_features = self.tabular_encoder(
            tabular
        )

        # Fusion
        combined = torch.cat(
            [image_features, tabular_features],
            dim=1
        )

        # Classification
        logits = self.classifier(combined)

        return logits


# ============================================================
# 3. CHARGEMENT DES DONNEES
# ============================================================

def load_data(root_dir, batch_size=8, num_workers=2):

    transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Charger le train en premier
    train_json = os.path.join(
        root_dir, "train", "index.json"
    )

    with open(train_json, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    train_df = pd.json_normalize(
        list(train_data.values())
    )

    columns = ["Fovea_X", "Fovea_Y"]

    tabular_mean = train_df[columns].astype(float).mean().values

    tabular_std = train_df[columns].astype(float).std().values

    tabular_std = np.where(
        tabular_std == 0, 1.0, tabular_std
    )

    # Dataset train
    train_dataset = RefugeDataset(
        root_dir=root_dir,
        split="train",
        transform=transform,
        tabular_mean=tabular_mean,
        tabular_std=tabular_std
    )

    # Dataset validation
    val_dataset = RefugeDataset(
        root_dir=root_dir,
        split="val",
        transform=transform,
        tabular_mean=tabular_mean,
        tabular_std=tabular_std
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# ============================================================
# 4. FONCTION DE PERTE
# ============================================================

def get_loss_function(root_dir, device):

    train_json = os.path.join(
        root_dir, "train", "index.json"
    )

    with open(train_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(list(data.values()))

    counts = df["Label"].value_counts()

    count_0 = counts.get(0, 0)
    count_1 = counts.get(1, 0)

    total = count_0 + count_1

    weights = torch.tensor(
        [
            total / (2 * count_0),
            total / (2 * count_1)
        ],
        dtype=torch.float32
    ).to(device)

    return nn.CrossEntropyLoss(weight=weights)


# ============================================================
# 5. INITIALISATION DU MODELE
# ============================================================

def create_model(device=None):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = SegFormerGlaucoma(
        pretrained=True,
        num_classes=2,
        dropout=0.3
    )

    model = model.to(device)

    return model
