
# ============================================================
# train.py
# SegFormer-B0 + MLP
# Glaucoma Classification
# ============================================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import wandb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from model import (
    load_data,
    create_model,
    get_loss_function
)


# ============================================================
# CONFIGURATION
# ============================================================

api_key = os.environ.get("WANDB_API_KEY")

if api_key:
    wandb.login(key=api_key)
else:
    raise RuntimeError("WANDB_API_KEY missing")

CONFIG = {
    "root_dir": "/kaggle/input/ton-dataset",
    "batch_size": 8,
    "epochs": 40,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 7,
    "num_workers": 2,
    "seed": 42,
    "image_size": 224,
    "model": "SegFormer-B0 + MLP",
    "classes": ["No Glaucoma", "Glaucoma"]
}

CHECKPOINT_DIR = "/kaggle/working/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)


# ============================================================
# REPRODUCTIBILITE
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["seed"])


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# WEIGHTS & BIASES
# ============================================================

run = wandb.init(
    project="glaucoma-detection",
    name="segformer-b0-multimodal",
    config={
        **CONFIG,

        # Configuration matérielle
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "gpu_count": torch.cuda.device_count(),

        # Framework
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda
    },
    mode="online"
)
model = create_model(device)
wandb.watch(
    model,
    log="gradients",
    log_freq=100
)

criterion = get_loss_function(
    CONFIG["root_dir"],
    device
)

# ============================================================
# DATA
# ============================================================

train_loader, val_loader = load_data(
    CONFIG["root_dir"],
    batch_size=CONFIG["batch_size"],
    num_workers=CONFIG["num_workers"]
)

print("Train samples:", len(train_loader.dataset))
print("Validation samples:", len(val_loader.dataset))


# ============================================================
# MODEL
# ============================================================

model = create_model(device)
print("Device utilisé :", next(model.parameters()).device)

criterion = get_loss_function(
    CONFIG["root_dir"],
    device
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"]
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# Mixed precision pour economiser la GPU
use_amp = device.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(labels, predictions, probabilities):

    accuracy = accuracy_score(
        labels, predictions
    )

    precision = precision_score(
        labels, predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        labels, predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        labels, predictions,
        pos_label=1,
        zero_division=0
    )

    try:
        auc = roc_auc_score(
            labels, probabilities
        )
    except ValueError:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    total_loss = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, tabular, labels in loader:

        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            outputs = model(images, tabular)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        probabilities = torch.softmax(
            outputs.detach().float(),
            dim=1
        )[:, 1]

        predictions = outputs.detach().argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_probabilities
    )

    return avg_loss, metrics


# ============================================================
# VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion):

    model.eval()

    total_loss = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, tabular, labels in loader:

        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            outputs = model(images, tabular)
            loss = criterion(outputs, labels)

        total_loss += loss.item()

        probabilities = torch.softmax(
            outputs.float(),
            dim=1
        )[:, 1]

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_probabilities
    )

    return avg_loss, metrics, all_labels, all_predictions


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

best_f1 = -1.0
epochs_without_improvement = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "train_f1": [],
    "val_f1": [],
    "train_auc": [],
    "val_auc": []
}

for epoch in range(CONFIG["epochs"]):

    print("\n" + "=" * 55)
    print(f"EPOCH {epoch + 1}/{CONFIG['epochs']}")
    print("=" * 55)

    # Training
    train_loss, train_metrics = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )

    # Validation
    val_loss, val_metrics, val_labels, val_preds = validate(
        model,
        val_loader,
        criterion
    )

    # Scheduler
    scheduler.step(val_metrics["f1"])

    current_lr = optimizer.param_groups[0]["lr"]

    # History
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_f1"].append(train_metrics["f1"])
    history["val_f1"].append(val_metrics["f1"])
    history["train_auc"].append(train_metrics["auc"])
    history["val_auc"].append(val_metrics["auc"])

    # Console
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")

    print("\nValidation metrics:")
    for key, value in val_metrics.items():
        print(f"{key}: {value:.4f}")

    print("Learning rate:", current_lr)

    # W&B logging
    wandb.log({

        "epoch": epoch + 1,

        "train/loss": train_loss,
        "train/accuracy": train_metrics["accuracy"],
        "train/precision": train_metrics["precision"],
        "train/recall": train_metrics["recall"],
        "train/f1": train_metrics["f1"],
        "train/auc": train_metrics["auc"],

        "val/loss": val_loss,
        "val/accuracy": val_metrics["accuracy"],
        "val/precision": val_metrics["precision"],
        "val/recall": val_metrics["recall"],
        "val/f1": val_metrics["f1"],
        "val/auc": val_metrics["auc"],

        "learning_rate": current_lr

    })

    # Save best model
    if val_metrics["f1"] > best_f1:

        best_f1 = val_metrics["f1"]
        epochs_without_improvement = 0

        torch.save({

            "epoch": epoch + 1,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "best_f1": best_f1,

            "config": CONFIG

        }, BEST_MODEL_PATH)

        print("\nBest model saved!")

        wandb.run.summary["best_val_f1"] = best_f1
        wandb.run.summary["best_epoch"] = epoch + 1

    else:

        epochs_without_improvement += 1

    # Early stopping
    if epochs_without_improvement >= CONFIG["patience"]:

        print("\nEarly stopping activated!")
        break


# ============================================================
# FINAL TRAINING GRAPHS
# ============================================================

import matplotlib.pyplot as plt

epochs_range = range(1, len(history["train_loss"]) + 1)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Loss
axes[0].plot(
    epochs_range,
    history["train_loss"],
    label="Train"
)

axes[0].plot(
    epochs_range,
    history["val_loss"],
    label="Validation"
)

axes[0].set_title("Training and Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

# F1
axes[1].plot(
    epochs_range,
    history["train_f1"],
    label="Train F1"
)

axes[1].plot(
    epochs_range,
    history["val_f1"],
    label="Validation F1"
)

axes[1].set_title("F1 Score")
axes[1].set_xlabel("Epoch")
axes[1].legend()

# AUC
axes[2].plot(
    epochs_range,
    history["train_auc"],
    label="Train AUC"
)

axes[2].plot(
    epochs_range,
    history["val_auc"],
    label="Validation AUC"
)

axes[2].set_title("AUC")
axes[2].set_xlabel("Epoch")
axes[2].legend()

plt.tight_layout()

plt.savefig(
    "/kaggle/working/training_curves.png",
    dpi=150
)

wandb.log({
    "training_curves": wandb.Image(fig)
})

plt.close(fig)

print("\nTraining completed!")
print("Best validation F1:", best_f1)
print("Model:", BEST_MODEL_PATH)

wandb.finish()
