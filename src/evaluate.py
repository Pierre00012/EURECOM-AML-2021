
# ============================================================
# evaluate.py
# Evaluation finale du modele SegFormer + MLP
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import torch
import wandb

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

from torch.utils.data import DataLoader

from model import (
    RefugeDataset,
    create_model
)

from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = "/kaggle/input/eurecom-aml-2021-challenge-2/refuge_data"

MODEL_PATH = (
    "/kaggle/working/checkpoints/best_model.pth"
)

OUTPUT_DIR = "/kaggle/working/evaluation"

BATCH_SIZE = 8
NUM_WORKERS = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# LOAD TRAIN STATISTICS
# ============================================================

train_json = os.path.join(
    ROOT_DIR,
    "train",
    "index.json"
)

with open(train_json, "r", encoding="utf-8") as f:
    train_data = json.load(f)

train_df = pd.json_normalize(
    list(train_data.values())
)

columns = ["Fovea_X", "Fovea_Y"]

tabular_mean = (
    train_df[columns].astype(float).mean().values
)

tabular_std = (
    train_df[columns].astype(float).std().values
)

tabular_std = np.where(
    tabular_std == 0,
    1.0,
    tabular_std
)


# ============================================================
# TEST DATASET
# ============================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])

test_dataset = RefugeDataset(

    root_dir=ROOT_DIR,

    split="test",

    transform=transform,

    tabular_mean=tabular_mean,

    tabular_std=tabular_std

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True

)

print("Test samples:", len(test_dataset))


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model = create_model(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully!")
print("Best epoch:", checkpoint["epoch"])
print("Best validation F1:", checkpoint["best_f1"])


# ============================================================
# WEIGHTS & BIASES
# ============================================================

run = wandb.init(

    project="glaucoma-detection",

    name="segformer-b0-final-evaluation",

    job_type="evaluation"

)


# ============================================================
# PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []

use_amp = device.type == "cuda"

with torch.no_grad():

    for images, tabular, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        tabular = tabular.to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            outputs = model(images, tabular)

        probabilities = torch.softmax(
            outputs.float(),
            dim=1
        )[:, 1]

        predictions = outputs.argmax(dim=1)

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    pos_label=1,
    zero_division=0
)

try:

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    average_precision = average_precision_score(
        all_labels,
        all_probabilities
    )

except ValueError:

    auc = float("nan")
    average_precision = float("nan")


print("\n========== TEST RESULTS ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print(f"PR-AUC:    {average_precision:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    all_labels,

    all_predictions,

    labels=[0, 1],

    target_names=[
        "No Glaucoma",
        "Glaucoma"
    ],

    zero_division=0

)

print("\nClassification Report:\n")
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(7, 6))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[
        "No Glaucoma",
        "Glaucoma"
    ],

    yticklabels=[
        "No Glaucoma",
        "Glaucoma"
    ]

)

plt.title("Confusion Matrix - Test Set")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=150
)

wandb.log({
    "test/confusion_matrix": wandb.Image(plt.gcf())
})

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

if len(np.unique(all_labels)) == 2:

    fpr, tpr, thresholds = roc_curve(
        all_labels,
        all_probabilities
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        color="blue",
        label=f"ROC AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve - Test Set")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "roc_curve.png"),
        dpi=150
    )

    wandb.log({
        "test/roc_curve": wandb.Image(plt.gcf())
    })

    plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

if len(np.unique(all_labels)) == 2:

    pr_precision, pr_recall, thresholds = (
        precision_recall_curve(
            all_labels,
            all_probabilities
        )
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        pr_recall,
        pr_precision,
        color="green",
        label=f"PR AUC = {average_precision:.4f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "precision_recall_curve.png"
        ),
        dpi=150
    )

    wandb.log({
        "test/precision_recall_curve": wandb.Image(
            plt.gcf()
        )
    })

    plt.close()


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(

    all_probabilities[all_labels == 0],

    bins=20,

    alpha=0.6,

    label="No Glaucoma"

)

plt.hist(

    all_probabilities[all_labels == 1],

    bins=20,

    alpha=0.6,

    label="Glaucoma"

)

plt.xlabel("Predicted probability of glaucoma")
plt.ylabel("Number of images")

plt.title("Prediction Probability Distribution")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "probability_distribution.png"
    ),
    dpi=150
)

wandb.log({
    "test/probability_distribution": wandb.Image(
        plt.gcf()
    )
})

plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1),

    "roc_auc": float(auc),

    "average_precision": float(average_precision),

    "confusion_matrix": cm.tolist(),

    "classification_report": report

}

results_df = pd.DataFrame(
    [results]
)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "metrics.csv"),
    index=False
)

with open(
    os.path.join(OUTPUT_DIR, "results.json"),
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# W&B FINAL LOGGING
# ============================================================

wandb.log({

    "test/accuracy": accuracy,

    "test/precision": precision,

    "test/recall": recall,

    "test/f1_score": f1,

    "test/roc_auc": auc,

    "test/average_precision": average_precision

})

wandb.run.summary["test_accuracy"] = accuracy
wandb.run.summary["test_precision"] = precision
wandb.run.summary["test_recall"] = recall
wandb.run.summary["test_f1"] = f1
wandb.run.summary["test_auc"] = auc

wandb.finish()


print("\n================================")
print("EVALUATION COMPLETED")
print("================================")

print("Results saved in:", OUTPUT_DIR)
print("W&B dashboard updated!")
